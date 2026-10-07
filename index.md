---
title: "Efficient Computing for AI"
subtitle: "LAB Material and Assignments"
---

Welcome to the course website.

This site is intentionally **Markdown-first**. Source material is stored in GitHub, rendered by [Quarto](https://quarto.org/), and published automatically to GitHub Pages.

## Course at a glance

::: {.columns}
::: {.column width="33%"}
### Lectures

Concepts, examples, and background material.

[Go to lectures →](lectures/index.md)
:::

::: {.column width="33%"}
### Labs

Hands-on assignments with reproducible setup instructions.

[Go to labs →](labs/index.md)
:::

::: {.column width="33%"}
### Resources

Software setup, references, and supporting material.

[Go to resources →](resources/software.md)
:::
:::

## How this site is built

```text
Markdown / Quarto
       ↓
     Quarto
       ↓
 GitHub Actions
       ↓
  GitHub Pages
```

The public repository should contain course material only. Keep instructor-only solutions, grading material, and private notes in a separate private repository.

::: {.callout-note}
## Local preview

Install [Quarto](https://quarto.org/docs/get-started/) and run:

```bash
quarto preview
```
:::
