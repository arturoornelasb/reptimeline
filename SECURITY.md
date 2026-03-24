# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in reptimeline, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email **arturoornelas62@gmail.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within 48 hours.
4. A fix will be developed privately and released as a patch.

## Scope

reptimeline is a research analysis library that processes model checkpoints and discrete representations locally. It does not run network services, handle authentication, or store user credentials. Security concerns are most likely to involve:

- Arbitrary code execution via maliciously crafted input files (JSON snapshots, pickle checkpoints)
- Dependency vulnerabilities in numpy, matplotlib, or optional dependencies (torch, plotly)

## Best Practices

- Only load model checkpoints and snapshot files from trusted sources.
- Keep dependencies up to date: `pip install --upgrade reptimeline`.
- When using the `model_loader` callback in extractors, ensure the loading function does not execute untrusted code.
