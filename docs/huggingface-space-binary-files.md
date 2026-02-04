# Pushing to Hugging Face Spaces: Binary File Rejection

## The Error

When pushing to a Hugging Face Space you may see:

```
remote: Your push was rejected because it contains binary files.
remote: Please use https://huggingface.co/docs/hub/xet to store binary files.
remote:
remote: Offending files:
remote:   - assets/some-image.jpg (ref: refs/heads/main)
```

**Cause:** Hugging Face does not accept raw binary files (e.g. PNG, JPG) in the Git pack. They must be stored via **Git LFS** or **Xet**. If the file was committed _before_ you added LFS tracking in `.gitattributes`, it stays as a normal blob in history and the push is rejected.

## Prerequisites

- **Git LFS** installed: [git-lfs.com](https://git-lfs.com)
  - Check: `git lfs version`

- `.gitattributes` already tracking images, e.g.:

  ```
  *.png filter=lfs diff=lfs merge=lfs -text
  *.jpg filter=lfs diff=lfs merge=lfs -text
  ```

## Solution: Migrate Existing Binaries to LFS

Rewrite history so all matching files are stored as LFS objects instead of raw blobs:

```bash
git lfs install
git lfs migrate import --include="*.jpg,*.png" --everything
git push space main --force
```

- `--include` — file patterns to migrate (adjust if you use other binary types).
- `--everything` — rewrite all branches so every commit uses LFS for those files.

After this, the push sends LFS pointers + LFS objects, not raw binaries, so the Space accepts it.

## “Your branch might have been rebased” when pulling

After running `git lfs migrate import`, you may see:

```
It looks like the current branch "main" might have been rebased.
Are you sure you still want to pull into it?
```

**Why it appears:** The migrate command rewrote history (new commit hashes). Your local `main` has the new history; the remote (e.g. `origin`) still has the old history. Git sees the divergence and assumes a rebase, so it warns that pulling might merge two different histories.

**Is it safe?** Yes. The warning is correct — your branch was effectively “rebased” (history rewritten). You do **not** want to pull the old history back into your local branch; that would mix old (non-LFS) and new (LFS) history and can cause confusion or re-introduce the binary blobs.

**What to do:**

- **Do not pull** (or cancel the pull) if the goal is to keep the LFS-migrated history as the source of truth.
- To sync other remotes (e.g. GitHub) with your rewritten history, force push to them so they match your local branch:
  ```bash
  git push origin main --force
  ```
- After that, local and remotes are in sync and you can pull/push normally again.

So: treat the warning as expected after a history rewrite; keep your migrated history and update remotes with a force push rather than pulling the old history back.

## If You Haven’t Set Up the Space Remote Yet

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

To fix a wrong URL:

```bash
git remote set-url space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

## Force Push After Creating the Space First

If you created the Space in the UI first (with default README/.gitattributes), your local history won’t match. Overwrite the remote with your local repo:

```bash
git push space main --force
```

## Optional: Silencing the “Repo Card” Warning

You may see:

```
Warning: empty or missing yaml metadata in repo card
```

This is optional. To fix it, add metadata to your Space (e.g. in the Space’s README or a config file). See [Spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference).

## Summary

| Situation                                | Action                                                                                                |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Push rejected: “contains binary files”   | Run `git lfs migrate import --include="*.jpg,*.png" --everything`, then `git push space main --force` |
| “Branch might have been rebased” on pull | Don’t pull; force push to other remotes instead: `git push origin main --force`                       |
| Wrong Space URL                          | `git remote set-url space <new-url>`                                                                  |
| Remote has different initial commit      | `git push space main --force` (overwrites remote)                                                     |
