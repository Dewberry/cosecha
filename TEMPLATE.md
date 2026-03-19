# Using This Template

> **Delete this file** once you have finished setting up your new project.

## 1. Install prerequisites

### pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### GitHub CLI (`gh`)

See the [official installation instructions](https://github.com/cli/cli#installation) for
your platform.

### Authenticate with GitHub

```bash
gh auth login
```

Follow the prompts — choose **GitHub.com**, **SSH**, and **Login with a web browser**.
`gh` will generate an SSH key and upload it to your GitHub account automatically if you
don't already have one. Verify the session:

```bash
gh auth status
```

---

## 2. Create a new repo from this template

```bash
gh repo create my-new-project \
  --template dewberry/template-pixi-project \
  --private \
  --clone
cd my-new-project
```

## 3. Find & replace all placeholders

Run these substitutions across the whole repo (your editor's global find-and-replace
works well here):

| Placeholder | Replace with | Example |
|---|---|---|
| `package-name` | PyPI / kebab-case name | `my-project` |
| `package_name` | Python / snake_case name | `my_project` |
| `PackageName` | Display name | `MyProject` |
| `A short description of the project.` | One-line description | `Does something great.` |
| `dewberry/package-name` | `<github-org>/<repo>` | `dewberry/my-project` |
| `dewberry.github.io/package-name` | Docs URL | `dewberry.github.io/my-project` |

Files that need updating:

- `pyproject.toml` — `name`, `description`, `urls`, `[tool.coverage.run].source_pkgs`,
  `[tool.ruff.lint.isort].known-first-party`, `[tool.ty.src].include`,
  `[tool.pixi.pypi-dependencies]`
- `mkdocs.yml` — `site_name`, `site_url`, `repo_url`, `edit_uri`, `extra.homepage`, `extra.social`
- `src/package_name/` — rename the directory itself to your snake_case name
- `src/package_name/__init__.py` — update the module docstring and imports
- `src/package_name/package_name.py` — rename and replace with your actual code
- `README.md` — fill in the project description and usage
- `CHANGELOG.md` — update the header
- `docs/examples/example.py` — rename and replace with real examples (see step 5)

## 4. Set up the development environment

Install the two environments you need locally:

```bash
# IDE support, running notebooks, and interactive development
pixi install -e dev

# Type checking with ty
pixi install -e typecheck
```

All other environments (`test311`, `test314`, `lint`, `docs`) are used on-demand or in
CI and do not need to be pre-installed.

### VS Code — ty extension

This project uses [ty](https://github.com/astral-sh/ty) for type checking. Install the
official VS Code extension for inline diagnostics:

```bash
code --install-extension astral-sh.ty
```

Or search for **"ty"** by Astral in the Extensions panel (`Ctrl+Shift+X` /
`Cmd+Shift+X`). The `pyproject.toml` already points ty at `.pixi/envs/typecheck/` via
`tool.ty.environment.python`, so no manual interpreter selection is needed.

### Install pre-commit hooks

```bash
pixi r pcupdate  # optionally bump all hooks to latest versions first
pixi r lint      # installs hooks and runs them across the repo
```

### Conventional Commits

This template is set up for [Conventional Commits](https://www.conventionalcommits.org).
All commit messages should follow the `<type>: <description>` format — e.g.
`feat: add raster export`, `fix: handle missing CRS`, `chore: bump dependencies`.
See `CONTRIBUTING.md` for the full type reference and changelog mapping.

## 5. Manage example notebooks

Notebooks live in `docs/examples/` as **percent-format `.py` files** (the source of
truth) paired with `.ipynb` files (generated, used by mkdocs-jupyter).

| Task | Command | When to use |
|---|---|---|
| Sync `.py` → `.ipynb` | `pixi r nb-sync` | After editing a `.py` notebook |
| Execute all notebooks | `pixi r nb-execute` | Before building docs |
| Sync then execute | `pixi r nb-run` | Typical pre-commit / pre-docs workflow |
| Pair an imported `.ipynb` | `pixi r nb-pair` | One-time setup for an external notebook |
| Strip outputs | `pixi r nb-clear` | Before committing (keeps diffs clean) |

**Adding a new notebook:**

1. Copy `docs/examples/example.py` and rename it.
2. Add a thumbnail at `docs/examples/images/<name>.svg` (or `.png`).
3. Add a card to `docs/examples/index.md` following the existing pattern.
4. Add a nav entry in `mkdocs.yml` under `Examples`.
5. Run `pixi r nb-run` to generate and execute the `.ipynb`.

## 6. Common development tasks

```bash
# Run tests (excluding network calls)
pixi r test

# Run all tests including network calls
pixi r test-all

# Type check
pixi r typecheck

# Lint & format (runs pre-commit on all files)
pixi r lint

# Build docs
pixi r docs-build

# Serve docs locally
pixi r docs-serve
```

## 7. Configure CI/CD secrets

Add these secrets in your GitHub repo settings (`Settings → Secrets → Actions`):

| Secret | Purpose |
|---|---|
| `CODECOV_TOKEN` | Coverage uploads to [codecov.io](https://codecov.io) |
| `PyPI` (trusted publisher) | PyPI release (configure via PyPI's trusted publisher UI — no token needed) |

## 8. Set up GitHub Pages for docs

After the first push to `main`, the `docs.yml` workflow will publish the site to the
`gh-pages` branch. Enable Pages in your repo settings:

`Settings → Pages → Source → Deploy from branch → gh-pages / root`

## 9. Tag a release

First update `CHANGELOG.md` with all commits since the last release:

```bash
pixi r changelog-update
git add CHANGELOG.md && git commit -m "chore: update changelog for v0.1.0"
git push
```

Then tag and publish:

```bash
pixi r release --version v0.1.0
```

This creates an annotated tag, pushes any unpushed commits first, then pushes the tag.
It is equivalent to:

```bash
git tag -a v0.1.0 -m v0.1.0 && git push && git push --tags
```

This triggers the `release.yml` workflow, which:

1. Generates release notes from conventional commits using [git-cliff](https://git-cliff.org)
2. Creates a GitHub Release
3. Builds and inspects the package (with SLSA provenance)
4. Publishes to PyPI
