<!--
Base branch: develop. Feature work merges into develop by squash; main only
takes release PRs from develop and hotfix/* (a PR into main from any other
branch is retargeted at develop by the Release source check).
-->

## What

## Why

## How to test

## Checklist

- [ ] Base is `develop` (or this is a release / hotfix PR into `main`)
- [ ] CI is green: sidecar tests, UI typecheck/tests/build, dev builds
- [ ] Tracker item linked, or "no tracker item"
