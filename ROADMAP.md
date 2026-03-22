# Roadmap

What's needed to take reptimeline from research prototype to production-ready, commercially viable tool.

## Técnico

### Distribución y CI
- [ ] Publicar en PyPI (el README dice `pip install reptimeline` pero no está publicado)
- [ ] GitHub Actions: tests, linting (ruff), coverage en cada PR
- [ ] Pre-commit hooks (ruff, type checking)
- [ ] Coverage reporting (pytest-cov está en deps pero no hay config ni badge)
- [ ] Badges en README: PyPI version, tests, coverage, Python versions

### Calidad de código
- [ ] Agregar type checking con mypy o pyright (hay type hints pero no se verifican)
- [ ] Logging estructurado (actualmente todo es print/stdout)
- [ ] Progress bars para operaciones largas (discovery triádica es O(K³))
- [ ] Manejo de errores más granular: excepciones custom en vez de ValueError genérico

### Documentación
- [ ] Generar sitio de docs con pdoc3 (ya está en optional deps, falta ejecutarlo)
- [ ] Publicar docs en GitHub Pages o Read the Docs
- [ ] Docstrings de API reference para todos los parámetros de thresholds en BitDiscovery
- [ ] Guía de migración para usuarios que vienen de triadic-microgpt

### Funcionalidad
- [ ] Soporte para snapshots incrementales (actualmente requiere todos los checkpoints en memoria)
- [ ] Export a formatos estándar: CSV, Parquet, WandB, TensorBoard
- [ ] Serialización robusta: versión de schema en los JSON de salida
- [ ] Paralelización de discovery triádica (actualmente single-thread, O(K³))
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
- [ ] Validar en más backends reales: VQ-VAE en producción, SAE a escala, concept bottleneck
- [ ] Case studies documentados más allá de MNIST y Pythia-70M
- [ ] Benchmarks de rendimiento: tiempo de análisis vs tamaño de codebook

### Go-to-market
- [ ] Landing page del proyecto (actualmente solo el repo)
- [ ] Tutorial interactivo (notebook) que funcione en Google Colab
- [ ] Integración con ecosistema ML: WandB callback, Lightning hook, HuggingFace integration
- [ ] Paper publicado y citeable (actualmente en draft)

### Comunidad
- [ ] CONTRIBUTING.md con guía de contribución
- [ ] Issue templates en GitHub
- [ ] Ejemplos reproducibles que corran sin datos propietarios ni GPUs

## Bloqueos

### Críticos (bloquean producción)
1. **No está en PyPI.** El README dice `pip install reptimeline` pero el paquete no existe. Nadie puede instalarlo sin clonar el repo.
2. **Sin CI/CD.** No hay GitHub Actions. Los 212 tests pasan localmente pero no hay garantía en PRs ni releases.
3. **Resultado negativo en predicción.** Los features descubiertos son causalmente selectivos pero no mejoran predicción. Esto limita el argumento comercial de "interpretabilidad actionable".

### Importantes (bloquean escala)
4. **Solo validado en 2 backends.** MNIST (32-bit) y Pythia-70M (32K features). Para vender como "backend-agnostic" se necesitan más validaciones reales.
5. **Discovery triádica no escala.** O(K³) con K = bits activos. Para SAEs con miles de features activos, es prohibitivo sin paralelización o sampling.
6. **Sentinel features sin resolver.** 8/16 features SAE mostraron zero cross-activation. No se puede distinguir entre selectividad perfecta y artefacto de sparsity.

### Menores (bloquean polish)
7. **Sin docs site.** pdoc3 está en deps pero nunca se ejecutó.
8. **Sin logging.** Dificulta debugging en producción.
9. **JSON sin schema version.** Si el formato cambia en v0.2, no hay forma de detectar archivos v0.1.
