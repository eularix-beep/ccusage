//! Opt-in per-database parse cache (local patch, not upstream).
//!
//! Enabled only when `CCUSAGE_ANTIGRAVITY_CACHE_DIR` is set; otherwise every
//! database is parsed exactly as upstream does. The cache stores the output of
//! `parse_sqlite_file` for one database. Cross-database deduplication still
//! runs over every database's events on every invocation, so a cache hit changes
//! how the per-database events are obtained, never how they are combined.
//!
//! A cached entry is used only when the database's length, modification time and
//! full content hash (plus the same for its `-wal` file) all match. A fresh
//! parse is stored only when the fingerprint taken before the parse equals the
//! one taken after it, so a database written during the parse is never cached.
//! Any unreadable, corrupt or mismatched entry is treated as a miss.

use std::{
    env,
    fs::{self, File},
    hash::{DefaultHasher, Hasher},
    io::{self, Read},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};

use crate::{Result, cli::SharedArgs, debug_log};

use super::parser::{AntigravityUsageEvent, parse_sqlite_file};

pub(super) const ANTIGRAVITY_CACHE_DIR_ENV: &str = "CCUSAGE_ANTIGRAVITY_CACHE_DIR";
// Bump whenever the parser or the cached event shape changes meaning.
const CACHE_SCHEMA: &str = concat!("iss-antigravity-parse-cache-1:", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FileFingerprint {
    len: u64,
    modified_nanos: u128,
    content_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Fingerprint {
    database: FileFingerprint,
    wal: Option<FileFingerprint>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    schema: String,
    path: String,
    fingerprint: Fingerprint,
    events: Vec<AntigravityUsageEvent>,
}

fn file_fingerprint(path: &Path) -> io::Result<Option<FileFingerprint>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let metadata = file.metadata()?;
    let modified_nanos = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let mut hasher = DefaultHasher::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut len = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.write(&buffer[..read]);
        len += read as u64;
    }
    Ok(Some(FileFingerprint {
        len,
        modified_nanos,
        content_hash: hasher.finish(),
    }))
}

fn fingerprint(database: &Path) -> Option<Fingerprint> {
    let mut wal_name = database.as_os_str().to_owned();
    wal_name.push("-wal");
    Some(Fingerprint {
        database: file_fingerprint(database).ok()??,
        wal: file_fingerprint(Path::new(&wal_name)).ok()?,
    })
}

fn entry_path(cache_dir: &Path, canonical: &str) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    hasher.write(canonical.as_bytes());
    cache_dir.join(format!("{:016x}.json", hasher.finish()))
}

fn read_entry(path: &Path, canonical: &str, fingerprint: &Fingerprint) -> Option<Vec<AntigravityUsageEvent>> {
    let entry: CacheEntry = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    (entry.schema == CACHE_SCHEMA && entry.path == canonical && &entry.fingerprint == fingerprint)
        .then_some(entry.events)
}

fn write_entry(path: &Path, entry: &CacheEntry) -> io::Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec(entry)?)?;
    fs::rename(&temporary, path).inspect_err(|_| {
        let _ = fs::remove_file(&temporary);
    })
}

/// Parses one database, through the cache when it is enabled.
pub(super) fn parse_database(
    database: &Path,
    shared: &SharedArgs,
) -> Result<Vec<AntigravityUsageEvent>> {
    let Some(cache_dir) = env::var_os(ANTIGRAVITY_CACHE_DIR_ENV).map(PathBuf::from) else {
        return parse_sqlite_file(database);
    };
    let canonical = fs::canonicalize(database)
        .unwrap_or_else(|_| database.to_path_buf())
        .to_string_lossy()
        .into_owned();
    let entry_file = entry_path(&cache_dir, &canonical);
    let before = fingerprint(database);
    if let Some(before) = &before
        && let Some(events) = read_entry(&entry_file, &canonical, before)
    {
        return Ok(events);
    }

    let parsed = parse_sqlite_file(database)?;
    if let Some(before) = before
        && fingerprint(database).as_ref() == Some(&before)
    {
        let entry = CacheEntry {
            schema: CACHE_SCHEMA.to_string(),
            path: canonical,
            fingerprint: before,
            events: parsed.clone(),
        };
        if let Err(error) = fs::create_dir_all(&cache_dir).and_then(|()| write_entry(&entry_file, &entry)) {
            debug_log(shared, format!("Antigravity parse cache write skipped: {error}"));
        }
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use ccusage_test_support::{EnvVarsGuard, Fixture};

    use super::*;
    use crate::{
        LoadedEntry, PricingMap,
        parser::test_support::{UsageFixture, create_database, metadata_blob},
    };

    fn shared() -> SharedArgs {
        SharedArgs {
            json: true,
            offline: true,
            ..SharedArgs::default()
        }
    }

    fn database(path: &Path, response_id: &'static str, input_tokens: u64, seconds: u64) {
        create_database(
            path,
            &[(
                1,
                metadata_blob(
                    Some("gemini-3-pro"),
                    Some(UsageFixture {
                        input_tokens,
                        total_output_tokens: 40,
                        visible_output_tokens: 40,
                        response_id: Some(response_id),
                        ..UsageFixture::default()
                    }),
                    Some((seconds, 0)),
                    &[],
                ),
            )],
            &[],
            &[],
        );
    }

    fn summary(entries: &[LoadedEntry]) -> Vec<String> {
        entries.iter().map(|entry| format!("{entry:?}")).collect()
    }

    fn load(fixture: &Fixture, cache: Option<&Path>) -> Vec<String> {
        let _guard = EnvVarsGuard::set_many([
            (
                super::super::paths::ANTIGRAVITY_DATA_DIR_ENV,
                Some(OsString::from(fixture.path("data"))),
            ),
            (ANTIGRAVITY_CACHE_DIR_ENV, cache.map(OsString::from)),
        ]);
        summary(&super::super::load_entries(&shared(), &PricingMap::load_embedded()).unwrap())
    }

    fn cache_files(dir: &Path) -> Vec<PathBuf> {
        let mut files = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        files.sort();
        files
    }

    #[test]
    fn cached_load_equals_uncached_load_including_cross_database_dedup() {
        let fixture = Fixture::new();
        database(&fixture.path("data/conversations/a.db"), "shared", 20, 1_778_000_000);
        database(&fixture.path("data/conversations/b.db"), "shared", 25, 1_778_000_001);
        database(&fixture.path("data/conversations/c.db"), "own", 30, 1_778_000_002);
        let cache = fixture.path("cache");

        let upstream = load(&fixture, None);
        let cold = load(&fixture, Some(&cache));
        assert_eq!(cache_files(&cache).len(), 3);
        let warm = load(&fixture, Some(&cache));

        assert_eq!(upstream.len(), 2, "shared response id must deduplicate");
        assert_eq!(cold, upstream);
        assert_eq!(warm, upstream);
    }

    #[test]
    fn changed_database_invalidates_its_entry() {
        let fixture = Fixture::new();
        let path = fixture.path("data/conversations/a.db");
        database(&path, "one", 20, 1_778_000_000);
        let cache = fixture.path("cache");
        let first = load(&fixture, Some(&cache));

        fs::remove_file(&path).unwrap();
        database(&path, "one", 99, 1_778_000_000);
        let changed = load(&fixture, Some(&cache));

        assert_ne!(changed, first);
        assert_eq!(changed, load(&fixture, None));
    }

    #[test]
    fn corrupt_or_foreign_entries_are_misses() {
        let fixture = Fixture::new();
        database(&fixture.path("data/conversations/a.db"), "one", 20, 1_778_000_000);
        let cache = fixture.path("cache");
        let upstream = load(&fixture, None);
        let _ = load(&fixture, Some(&cache));
        let entry = cache_files(&cache).remove(0);

        fs::write(&entry, b"{not json").unwrap();
        assert_eq!(load(&fixture, Some(&cache)), upstream);

        let mut foreign: serde_json::Value = serde_json::from_slice(&fs::read(&entry).unwrap()).unwrap();
        foreign["schema"] = "iss-antigravity-parse-cache-0:old".into();
        foreign["events"] = serde_json::json!([]);
        fs::write(&entry, serde_json::to_vec(&foreign).unwrap()).unwrap();
        assert_eq!(load(&fixture, Some(&cache)), upstream);

        foreign["schema"] = CACHE_SCHEMA.into();
        foreign["path"] = "elsewhere.db".into();
        fs::write(&entry, serde_json::to_vec(&foreign).unwrap()).unwrap();
        assert_eq!(load(&fixture, Some(&cache)), upstream);
    }

    #[test]
    fn deleted_database_drops_out_exactly_as_upstream() {
        let fixture = Fixture::new();
        database(&fixture.path("data/conversations/a.db"), "one", 20, 1_778_000_000);
        database(&fixture.path("data/conversations/b.db"), "two", 30, 1_778_000_001);
        let cache = fixture.path("cache");
        let _ = load(&fixture, Some(&cache));

        fs::remove_file(fixture.path("data/conversations/b.db")).unwrap();

        assert_eq!(load(&fixture, Some(&cache)), load(&fixture, None));
        assert_eq!(load(&fixture, None).len(), 1);
    }

    #[test]
    fn wal_file_is_part_of_the_fingerprint() {
        let fixture = Fixture::new();
        let path = fixture.path("data/conversations/a.db");
        database(&path, "one", 20, 1_778_000_000);
        let without = fingerprint(&path).unwrap();
        fs::write(fixture.path("data/conversations/a.db-wal"), b"x").unwrap();
        let with = fingerprint(&path).unwrap();

        assert!(without.wal.is_none());
        assert_ne!(without, with);
    }
}
