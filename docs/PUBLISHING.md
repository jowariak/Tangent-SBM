# Publishing

This folder is prepared as a local Git repository. No remote repository URL is
claimed until GitHub creation and upload succeed. Select an appropriate license
for the authors' original code with the coauthors; upstream licenses are already
preserved and must remain intact.

To publish using GitHub CLI, after installing it and signing in:

```bash
git add .
git commit -m "Initial research code release"
gh repo create tangent-sbm --public --source=. --remote=origin --push
```

Use an explicit `OWNER/tangent-sbm` in place of `tangent-sbm` for an organization.
The repository is public only after that command succeeds. Do not claim in the
paper that code is publicly available before there is a working link.

For an anonymous submission, use the conference-approved anonymous supplementary
code route instead of linking an identifying personal repository directly.
The public package is not an anonymized archive of the authors' identity.
