# EGAgent project page (gh-pages)

This folder is the source for the GitHub project page, mimicking [LaViLa's gh-pages](https://github.com/facebookresearch/LaViLa/tree/gh-pages).

## Setup

1. **Replace repo URL**  
   In `index.html`, replace every `YOUR_GITHUB_ORG_OR_USERNAME` with your GitHub username or organization (e.g. `aniket` or `yourlab`).

2. **Optional: paper thumbnail**  
   Add a screenshot of the paper (e.g. first page or a figure) as `assets/paper-screenshot.png` (recommended ~150px width). If you omit it, the Paper section will show a broken image until you add it.

3. **Figures in main branch**  
   The page loads teaser and pipeline images from the **main** branch via raw URLs, e.g.  
   `https://github.com/YOUR_ORG/egagent/raw/main/figs/egagent_teaser.png`.  
   Ensure `figs/egagent_teaser.png`, `figs/egagent_pipeline.png`, and (if used) `figs/egolifeqa_categorywise_accuracy.png` exist in the repo root on `main`.

4. **Author links & affiliations**  
   Edit the author table and affiliations in `index.html` to add personal/homepage links and correct institution names/numbers.

## Publishing

- **Option A – gh-pages branch**  
  Push this folder to a branch named `gh-pages`:
  ```bash
  git subtree push --prefix gh-pages origin gh-pages
  ```
  Or copy the contents of `gh-pages/` into a new branch and push that branch.  
  GitHub will serve it at `https://YOUR_ORG.github.io/egagent/`.

- **Option B – GitHub Pages from branch**  
  In the repo: **Settings → Pages → Source**: choose branch `gh-pages` and root (or the folder that contains `index.html`).

## Files

- `index.html` – main project page
- `assets/style.css` – styles (from LaViLa template)
- `assets/paper-screenshot.png` – optional paper thumbnail (add yourself)
- `assets/bib.txt` – optional plain-text BibTeX (in-page BibTeX is already in `index.html`)
