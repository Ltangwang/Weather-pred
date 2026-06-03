# Uploading to GitHub

The repository is committed locally on branch `master`. Follow these steps on your machine (or this server) to publish.

## 1. Create an empty repository on GitHub

- Go to https://github.com/new
- Name: e.g. `Weather-pred` or `ProbWrapper-WeatherBench`
- **Do not** initialize with README (this repo already has one)
- Copy the HTTPS or SSH URL, e.g. `https://github.com/YOUR_USER/Weather-pred.git`

## 2. Set commit identity (once per machine)

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

(Optional) Amend the last commit author after setting identity:

```bash
git commit --amend --reset-author --no-edit
```

## 3. Add remote and push

```bash
cd /path/to/Weather-pred
git remote add origin https://github.com/YOUR_USER/Weather-pred.git
git push -u origin master
```

For SSH:

```bash
git remote add origin git@github.com:YOUR_USER/Weather-pred.git
git push -u origin master
```

## 4. What is **not** pushed (by design)

See `.gitignore`: checkpoints, logs, `results/`, `*.zip`, local summary packs, `.cursor/`, and WeatherBench `.nc` data.

Clone OpenSTL separately and download data per [WORKFLOW.md](WORKFLOW.md).

## 5. Verify after push

- README renders on the repo home page
- `examples/smoke_probabilistic.sh` is visible
- `docs/WORKFLOW.md` link works
