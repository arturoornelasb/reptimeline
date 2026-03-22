# Roadmap

What's needed to take reptimeline from research prototype to production-ready, commercially viable tool.

## Técnico

### Distribución y CI
- [ ] Publicar en PyPI (el README dice `pip install reptimeline` pero no está publicado)
- [x] GitHub Actions: tests (Python 3.10-3.13), ruff lint, coverage en cada PR
- [x] Pre-commit hooks (.pre-commit-config.yaml con ruff check + ruff-format)
- [x] Coverage reporting (pytest-cov configurado en pyproject.toml + coverage XML en CI)
- [x] Badges en README: CI status, Python versions, license

### Calidad de código
- [x] Codebase ruff-clean (0 warnings: import sorting, unused imports, line length, naming)
- [x] Type checking con mypy (0 errors, integrado en CI)
- [x] Logging estructurado en CLI y extractors (print_summary/print_report siguen como API de usuario)
- [x] Progress bars con tqdm en extract_sequence y discovery triádica
- [x] Manejo de errores más granular: excepciones custom (SnapshotError, ExtractionError, DiscoveryError, ConfigurationError)

### Documentación
- [x] Generar sitio de docs con pdoc3 + GitHub Pages workflow automático
- [x] Docstrings de API reference para todos los parámetros de thresholds en BitDiscovery
- [ ] Guía de migración para usuarios que vienen de triadic-microgpt

### Funcionalidad
- [ ] Soporte para snapshots incrementales (actualmente requiere todos los checkpoints en memoria)
- [ ] Export a formatos estándar: CSV, Parquet, WandB, TensorBoard
- [x] Serialización robusta: `schema_version: "0.1"` en ConceptSnapshot y Timeline JSON
- [ ] Paralelización de discovery triádica (actualmente single-thread, O(K³))
- [x] Extractors built-in: SAEExtractor, VQVAEExtractor, FSQExtractor (4 backends validados con tests)
- [ ] Más extractors: concept bottleneck models, DINO features, quantized LLMs
- [ ] Tests de rendimiento con datasets grandes (actual validación: 10-60 conceptos)

### Visualización
- [ ] Plots interactivos (Plotly o similar) además de matplotlib estático
- [ ] Colores de capas dinámicos en layer_emergence.py (actualmente hardcoded a 6)
- [ ] Export de plots a HTML standalone

## Comercial

### Licencia y legal
- [ ] Definir términos de licencia comercial (precio, tiers, límites)
- [ ] Página de pricing o contacto comercial (actualmente solo un email)
- [ ] CLA (Contributor License Agreement) si se aceptan contribuciones externas
- [ ] Revisar si BUSL-1.1 → AGPL-3.0 (2030) es el timeline correcto para el negocio

### Validación
- [ ] Resolver el resultado negativo en predicción (-0.13% embedding, -4.20% MLP vs baseline)
- [x] SAE validado con Pythia-70M (32K features); VQ-VAE y FSQ con unit tests
- [ ] Validar VQ-VAE y FSQ en producción real; validar concept bottleneck models
- [ ] Case studies documentados más allá de MNIST y Pythia-70M
- [ ] Benchmarks de rendimiento: tiempo de análisis vs tamaño de codebook

### Go-to-market
- [ ] Landing page del proyecto (actualmente solo el repo)
- [ ] Tutorial interactivo (notebook) que funcione en Google Colab
- [ ] Integración con ecosistema ML: WandB callback, Lightning hook, HuggingFace integration
- [ ] Paper publicado y citeable (actualmente en draft)

### Comunidad
- [x] CONTRIBUTING.md con guía de contribución
- [x] Issue templates en GitHub (bug report + feature request)
- [x] Ejemplos reproducibles que corran sin datos propietarios ni GPUs (default device=cpu, hardcoded paths removed)

## Bloqueos

### Críticos (bloquean producción)
1. **No está en PyPI.** `python -m build` ya produce sdist + wheel correctamente y el publish workflow existe. Falta: crear cuenta en PyPI y configurar trusted publishing.
2. **Resultado negativo en predicción.** Los features descubiertos son causalmente selectivos pero no mejoran predicción. Esto limita el argumento comercial de "interpretabilidad actionable".

### Importantes (bloquean escala)
3. **Discovery triádica no escala.** O(K³) con K = bits activos. Para SAEs con miles de features activos, es prohibitivo sin paralelización o sampling.
4. **Sentinel features sin resolver.** 8/16 features SAE mostraron zero cross-activation. No se puede distinguir entre selectividad perfecta y artefacto de sparsity.
5. **Validación en producción.** 4 extractors implementados con 212 tests, pero VQ-VAE y FSQ solo tienen unit tests — falta validación con modelos reales.

### Resueltos
- ~~Sin CI/CD~~ — GitHub Actions CI: tests (Python 3.10-3.13), ruff lint, coverage. Publish workflow con trusted publishing.
- ~~Solo 2 backends~~ — 4 extractors (SAE, VQ-VAE, FSQ + triadic example). SAE validado con Pythia-70M.
- ~~Sin logging~~ — CLI y extractors usan `logging` module.
- ~~JSON sin schema version~~ — `schema_version: "0.1"` en `ConceptSnapshot` y `Timeline`.
- ~~Ruff warnings~~ — Codebase 100% ruff-clean.
- ~~Sin type checking~~ — mypy 0 errors, integrado en CI.
- ~~Pre-commit hooks~~ — ruff check + ruff-format.
- ~~Sin progress bars~~ — tqdm en extract_sequence y discovery triádica.
- ~~Sin docs site~~ — pdoc3 + GitHub Pages deploy automático.
- ~~Sin CONTRIBUTING.md~~ — Guía de contribución + issue templates.
- ~~Build no verificado~~ — `python -m build` produce sdist + wheel correctamente.
- ~~ValueError genérico~~ — Excepciones custom: SnapshotError, ExtractionError, DiscoveryError, ConfigurationError.
- ~~Sin docstrings de thresholds~~ — BitDiscovery.__init__ y discover() documentados con rangos, defaults y guías prácticas.
- ~~Ejemplos requieren GPU~~ — Todos los examples/ usan default device=cpu; hardcoded paths eliminados.
