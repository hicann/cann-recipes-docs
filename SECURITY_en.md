# Security Policy

CANN Recipes Docs takes the security of its documentation, tools, and documentation website seriously. If you discover a security issue that might affect this repository or its users, report it privately by following this policy.

## Scope

This policy applies to the following content in the current default branch and the currently published documentation website:

- Technical documents and images under `infer/`, `train/`, `embodied_ai/`, and `cann_features/`.
- The static documentation website, index builder, and local development server under `utils/site/`.
- Documentation metadata validation tools under `utils/scripts/`.
- Governance documents and configuration files in the repository root.

This repository does not distribute CANN product binaries, model weights, or datasets. CANN components, third-party projects, and external services referenced by the documentation follow their own security policies. If an issue is located in another project, report it to that project first.

This repository does not maintain separate release branches. Security fixes apply to the current default branch and the currently published documentation website. If an issue exists only in a historical commit, first determine whether it still affects current content or users.

## Security Issue Scope

Report the following issues privately as security vulnerabilities:

- The repository or generated website unintentionally exposes access tokens, passwords, private keys, personal data, internal addresses, or other non-public information.
- A flaw in the documentation website or its scripts enables cross-site scripting, path traversal, arbitrary file access, command execution, or other unauthorized actions.
- Documentation, images, generated indexes, or external resource references can be exploited to affect visitors, build environments, or maintenance workflows.
- Instructions in the documentation create a reproducible security risk, such as credential exposure, a permissions boundary bypass, or unintended code execution.
- Any other issue for which you can clearly describe the attack prerequisites and security impact.

The following issues are generally not security vulnerabilities. Report them publicly through this repository's Issues page:

- Typographical, formatting, translation, broken-link, or general content accuracy issues.
- Feature requests, compatibility issues, performance issues, or usage questions that do not cross a security boundary.
- Issues that affect only unmaintained historical content and do not affect current users.

If you are unsure whether an issue is a security vulnerability, report it privately and let the CANN community security team assess it.

## Reporting a Vulnerability

Do not disclose an unpatched security issue in a public Issue, Pull Request, discussion, or any other public channel.

Send suspected security vulnerabilities to the CANN security SIG at <security@cann.osinfra.cn>. Because vulnerability information can be sensitive, use the CANN security team's [PGP public key](https://gitcode.com/cann/community/blob/master/security/PGP/CANN_public_key.asc) to encrypt your email. For the complete process, see the CANN community [Vulnerability Management Policy](https://gitcode.com/cann/community/blob/master/security/security.md) (Chinese).

To help the team confirm and reproduce the issue, provide as much of the following information as possible:

- The affected file path, commit ID, page URL, or website version.
- A description of the issue, attack prerequisites, expected behavior, and actual behavior.
- Minimal reproduction steps, proof-of-concept code, or a test sample.
- The potential impact on confidentiality, integrity, and availability.
- Sanitized logs, screenshots, or other evidence.
- A feasible remediation or mitigation proposal, if available.
- Your contact information and planned disclosure timeline, if applicable.

Provide only the data required to analyze the issue. Do not send real access tokens, passwords, private keys, data obtained without authorization, or unrelated personal information.

## Response and Disclosure

The CANN community vulnerability management policy specifies the following process:

- The CANN community security team receives, investigates, handles, and discloses relevant security vulnerabilities.
- The security team responds within 48 hours after receiving an email report and keeps the reporter informed of progress.
- The security team uses CVSS v3 and considers both impact and real-world likelihood when assigning a handling priority.
- The CANN community does not publicly disclose, discuss, or confirm a security issue before it completes the investigation, provides a fix, and publishes a security advisory.
- After a fix is available, the CANN community may publish a security advisory, a CVE, affected versions, and remediation information based on the scope of impact.

Keep vulnerability information confidential until the security team confirms that public disclosure is appropriate. Repository maintainers cooperate with the CANN community security team on reproduction, remediation, and documentation updates. For the team's responsibilities and membership, see the [CANN security SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/security/README.md) (Chinese). For published information, see the [CANN Security Advisories](https://gitcode.com/cann/community/blob/master/security/sa.md) (Chinese).

## Secure Use and Contribution Guidelines

### Least Privilege

- Do not use `root` or another administrator account to run scripts or the local website from this repository.
- Set `umask 0027` or a more restrictive value in the local working directory so that new directories have permissions no broader than `750` and new files have permissions no broader than `640`.
- Grant access to the repository, generated artifacts, logs, and publishing environment only to people who require it.

### Protect Sensitive Information

- Do not commit accounts, passwords, access tokens, private keys, certificate private keys, license files, internal network addresses, private image registries, local absolute paths, or logs and screenshots that contain sensitive information.
- Before committing, check documents, code blocks, image metadata, generated indexes, and Git history for sensitive information.
- `utils/site/content/search-index.json` contains searchable text from technical documents. Do not add information to source documents that must not become public.
- If sensitive information has already been committed, revoke or rotate the affected credentials immediately and report the exposure through the private channel described earlier. Deleting the text only from the current version does not remove it from Git history.

### Local Website and Scripts

- Before running a command from the documentation, inspect the command, script, download URL, and target path. Do not grant elevated privileges directly to content from an unknown source.
- `utils/site/build_content.py` reads files from this repository and generates website indexes. Run it only in a trusted working tree, and review the generated content before publishing it.
- `utils/site/serve.py` is a local development server that binds only to `127.0.0.1`. It is not a production-grade web server. Do not change its bind address and expose it directly to the internet or an untrusted network.
- The static website renders Markdown, syntax highlighting, formulas, and Mermaid diagrams in the browser. Treat new or modified renderable content as untrusted input, and check it for script injection, dangerous links, and unsafe HTML.

## Public Network Addresses and Third-Party Resources

This documentation repository contains external links to official CANN resources, related code repositories, and third-party projects. The index builder does not download the documents or images behind these links. When a browser opens a link or loads a remote resource, it connects directly to the corresponding third-party service.

The static website currently uses the following main public resources:

| Type | Location | Public Address or Domain | Purpose |
|---|---|---|---|
| Frontend resources | `utils/site/index.html` | `cdn.tailwindcss.com`, `cdn.jsdelivr.net` | Loads Tailwind CSS, Marked, highlight.js, KaTeX, and Mermaid. |
| Font resources | `utils/site/index.html` | `fonts.googleapis.com`, `fonts.gstatic.com` | Loads website fonts and icon fonts. |
| Sample images | `utils/site/index.html` | `images.unsplash.com` | Loads fallback sample images for the homepage. |
| Community and source links | Technical documents and the website | `gitcode.com` | Links to the CANN community, related repositories, source files, Issues, and contribution pages. |
| Documentation references | `infer/`, `train/`, `embodied_ai/`, `cann_features/` | External URLs declared in the documents | Provides entry points to product documentation, upstream projects, and technical resources. |

Before deploying the website, evaluate third-party resources against the deployment environment's security policy. Configure appropriate content security policies, network access controls, and resource integrity protection. Every new external resource must come from a trusted source, have a clear purpose, and undergo review as part of the change.
