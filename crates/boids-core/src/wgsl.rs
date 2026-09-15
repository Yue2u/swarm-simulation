//! Minimal WGSL source preprocessor.
//!
//! `wgpu` has no `#include`, but splitting shared math across files is essential once the same
//! `sdf()` field is used by the ocean raymarch pass, the terrain shading pass and the simulation
//! avoidance force. Rather than adding a build-script dependency, this module implements the one
//! directive we need:
//!
//! ```wgsl
//! //#include "common/layout.wgsl"
//! ```
//!
//! Rules, deliberately kept small so there is nothing to debug at 3am:
//!
//! * the directive must be on its own line, starting at column 0,
//! * paths are relative to the shader root (`<workspace>/shaders`),
//! * each file is included **at most once** per compilation unit, guarded by its resolved path,
//! * cycles are an error, not a hang,
//! * `#pragma once` is not supported because inclusion is already idempotent.
//!
//! It lives in `boids-core` rather than in the GPU crate so that both `boids-gpu` (simulation
//! shaders) and `boids-render` (render shaders) can share it without depending on each other, and
//! so it is unit testable with no GPU present.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Where a shader came from, for error messages.
pub type ShaderPath = PathBuf;

/// Errors raised while resolving includes.
#[derive(Debug)]
pub enum ShaderError {
    /// The root directory does not exist.
    MissingRoot(PathBuf),
    /// A file could not be read.
    Io {
        /// Path that failed.
        path: PathBuf,
        /// Underlying IO error.
        source: std::io::Error,
    },
    /// An include cycle was detected. The chain is reported in order.
    Cycle(Vec<PathBuf>),
    /// A `//#include` line used a path that escapes the shader root.
    EscapesRoot {
        /// The offending include target as written.
        target: String,
    },
    /// A `//#include` line is malformed.
    Malformed {
        /// File the directive was found in.
        path: PathBuf,
        /// 1-based line number.
        line: usize,
        /// The offending line.
        text: String,
    },
}

impl core::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingRoot(p) => write!(f, "shader root does not exist: {}", p.display()),
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Cycle(chain) => {
                write!(f, "include cycle:")?;
                for p in chain {
                    write!(f, " -> {}", p.display())?;
                }
                Ok(())
            }
            Self::EscapesRoot { target } => {
                write!(f, "include target escapes the shader root: {target}")
            }
            Self::Malformed { path, line, text } => write!(
                f,
                "malformed include at {}:{line}: {text}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ShaderError {}

/// A compiled WGSL compilation unit plus the provenance of everything that went into it.
///
/// Keeping `sources` around means a shader compilation error can be reported against the original
/// file names instead of a line number in an anonymous blob.
#[derive(Debug, Clone)]
pub struct CompiledShader {
    /// Fully expanded WGSL source.
    pub source: String,
    /// Resolved files that contributed, in inclusion order.
    pub sources: Vec<ShaderPath>,
}

/// Resolves `//#include` directives relative to a shader root directory.
#[derive(Debug, Clone)]
pub struct ShaderLoader {
    root: PathBuf,
}

impl ShaderLoader {
    /// Creates a loader rooted at `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The shader root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The default root: `<workspace>/shaders`, derived from this crate's manifest directory.
    ///
    /// `CARGO_MANIFEST_DIR` is baked in at compile time, so this works from tests and from the
    /// final binary alike, as long as the source tree is where it was built. That is the right
    /// trade for a simulation with a hot-tunable shader set: editing a `.wgsl` file and restarting
    /// is far faster than a rebuild, and there is no embed step to keep in sync.
    #[must_use]
    pub fn workspace_default() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("shaders");
        Self::new(root)
    }

    /// Compiles `relative` (e.g. `"sim/integrate.wgsl"`), inlining every include.
    ///
    /// # Errors
    /// Returns [`ShaderError`] if a file is missing, an include is malformed or cyclic, or an
    /// include escapes the shader root.
    pub fn compile(&self, relative: &str) -> Result<CompiledShader, ShaderError> {
        if !self.root.is_dir() {
            return Err(ShaderError::MissingRoot(self.root.clone()));
        }
        let entry = self.resolve(relative)?;
        let mut sources = Vec::new();
        let mut visited = BTreeSet::new();
        let mut stack = Vec::new();
        let source = self.expand(&entry, &mut visited, &mut stack, &mut sources)?;
        Ok(CompiledShader { source, sources })
    }

    /// Resolves a path relative to the root, rejecting anything that escapes it.
    fn resolve(&self, relative: &str) -> Result<PathBuf, ShaderError> {
        let rel = Path::new(relative);
        // Reject absolute paths and parent traversal outright instead of relying on canonicalize,
        // which would need the file to exist first.
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(ShaderError::EscapesRoot {
                target: relative.to_string(),
            });
        }
        Ok(self.root.join(rel))
    }

    fn expand(
        &self,
        path: &Path,
        visited: &mut BTreeSet<PathBuf>,
        stack: &mut Vec<PathBuf>,
        sources: &mut Vec<PathBuf>,
    ) -> Result<String, ShaderError> {
        if stack.contains(&path.to_path_buf()) {
            let mut chain = stack.clone();
            chain.push(path.to_path_buf());
            return Err(ShaderError::Cycle(chain));
        }
        let text = std::fs::read_to_string(path).map_err(|source| ShaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        stack.push(path.to_path_buf());
        sources.push(path.to_path_buf());

        let mut out = String::with_capacity(text.len() + 512);
        for (idx, line) in text.lines().enumerate() {
            match include_target(line) {
                Some(Ok(target)) => {
                    let resolved = self.resolve(target)?;
                    if visited.insert(resolved.clone()) {
                        let included = self.expand(&resolved, visited, stack, sources)?;
                        // A line marker makes shader compiler errors point at the included file.
                        let _ = writeln!(out, "// >>> {target}");
                        out.push_str(&included);
                        let _ = writeln!(out, "// <<< {target}");
                    } else {
                        let _ = writeln!(out, "// (already included) {target}");
                    }
                }
                Some(Err(())) => {
                    return Err(ShaderError::Malformed {
                        path: path.to_path_buf(),
                        line: idx + 1,
                        text: line.to_string(),
                    })
                }
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }

        stack.pop();
        Ok(out)
    }
}

/// Recognises an include directive line.
///
/// Returns `None` when the line is ordinary WGSL, `Some(Ok(target))` for a well-formed directive
/// and `Some(Err(()))` for a directive-shaped line that is malformed.
fn include_target(line: &str) -> Option<Result<&str, ()>> {
    let rest = line.strip_prefix("//#include")?;
    let rest = rest.trim();
    let (Some(inner), true) = (
        rest.strip_prefix('"').and_then(|r| r.strip_suffix('"')),
        rest.len() >= 2,
    ) else {
        return Some(Err(()));
    };
    if inner.is_empty() {
        Some(Err(()))
    } else {
        Some(Ok(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loader() -> ShaderLoader {
        ShaderLoader::workspace_default()
    }

    #[test]
    fn directive_parsing() {
        assert_eq!(include_target("//#include \"a.wgsl\""), Some(Ok("a.wgsl")));
        assert_eq!(include_target("  //#include \"a.wgsl\""), None);
        assert_eq!(include_target("// #include \"a.wgsl\""), None);
        assert_eq!(include_target("var a = 1;"), None);
        assert!(matches!(include_target("//#include a.wgsl"), Some(Err(()))));
        assert!(matches!(include_target("//#include \"\""), Some(Err(()))));
    }

    #[test]
    fn rejects_paths_escaping_the_root() {
        let l = loader();
        assert!(matches!(
            l.compile("../Cargo.toml"),
            Err(ShaderError::EscapesRoot { .. })
        ));
        assert!(matches!(
            l.compile("shaders/common/../../Cargo.toml"),
            Err(ShaderError::EscapesRoot { .. })
        ));
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let l = loader();
        assert!(matches!(
            l.compile("definitely/not/here.wgsl"),
            Err(ShaderError::Io { .. })
        ));
    }

    #[test]
    fn missing_root_is_reported_clearly() {
        let l = ShaderLoader::new("/nonexistent/shader/root");
        assert!(matches!(l.compile("x.wgsl"), Err(ShaderError::MissingRoot(_))));
    }

    #[test]
    fn workspace_root_exists_and_holds_the_shared_headers() {
        // Guards against a refactor moving the shader tree: the loader resolves relative to this
        // crate, so a moved directory would otherwise only fail at runtime, in the app.
        let l = loader();
        assert!(l.root().is_dir(), "shader root {} missing", l.root().display());
        for name in [
            "common/layout.wgsl",
            "common/math_common.wgsl",
            "common/sdf.wgsl",
        ] {
            let c = l.compile(name).unwrap_or_else(|e| panic!("compiling {name}: {e}"));
            assert!(!c.source.contains("//#include"), "{name} left an unresolved include");
            assert!(
                !c.sources.is_empty(),
                "{name} produced no provenance information"
            );
        }
    }

    #[test]
    fn includes_are_deduplicated() {
        // Compiling a file that includes the same header twice must inline it once: duplicate
        // WGSL declarations would be a compile error on the GPU.
        let l = loader();
        let c = l.compile("tests/dedup.wgsl").expect("dedup shader");
        let count = c.source.matches("fn dedup_probe").count();
        assert_eq!(count, 1, "header was inlined {count} times");
    }

    #[test]
    fn include_cycle_is_an_error() {
        let l = loader();
        assert!(matches!(
            l.compile("tests/cycle_a.wgsl"),
            Err(ShaderError::Cycle(_))
        ));
    }
}
