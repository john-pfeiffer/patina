# Repository rulesets

Import each file in GitHub: **Settings → Rules → Rulesets → New ruleset ▾ →
Import a ruleset**. Import `develop.json` and `main.json` after the `develop`
branch exists and is the default branch; import `release-tags.json` any time.

| File | Protects | Flow it enforces |
|---|---|---|
| `develop.json` | `develop` | feature/fix/chore/claude branches arrive by squash-merged PR with all CI jobs green and the branch up to date; no force pushes, no deletion, linear history |
| `main.json` | `main` | only merge-commit PRs whose source is `develop` or `hotfix/*` (the `Release source` check from `.github/workflows/release-source.yml`), all CI jobs green |
| `release-tags.json` | tags `v*` | only an admin can create a release tag; tags cannot be moved or deleted |

The bypass actor is the repository Admin role (`actor_id` 5) in
"pull requests only" mode for the branch rulesets, "always" for tags. If the
importer rejects the bypass entry, set `"bypass_actors": []`, import, then add
yourself under *Bypass list* in the ruleset editor.

Required-check names must match the job names in `.github/workflows/ci.yml`
exactly; change both together.
