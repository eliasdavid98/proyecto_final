# Ética operativa y criterios de retirada

## Alcance del análisis por subgrupo

El análisis usa el holdout etiquetado de PSet 3, separado con `random_state=42`,
y el champion de PSet 3 al umbral `0.50`. Ese holdout ya se usó para seleccionar
el champion; estos cortes son diagnósticos descriptivos posteriores a la
selección, no una auditoría independiente ni una garantía de desempeño futuro.
La etiqueta `exited` tampoco acredita un horizonte de 30–60 días.

| Grupo | n | Churn observado | Contactar | Sin contacto | Precision | Recall | F2 | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edad <40 | 1.932 | 7,19 % | 2,12 % | 97,88 % | 0,6585 | 0,1942 | 0,2261 | 0,4676 |
| Edad 40+ | 1.068 | 44,94 % | 45,79 % | 54,21 % | 0,7975 | 0,8125 | 0,8095 | 0,8991 |
| Mujer | 1.292 | 28,48 % | 26,16 % | 73,84 % | 0,7929 | 0,7283 | 0,7403 | 0,8591 |
| Hombre | 1.708 | 14,70 % | 11,24 % | 88,76 % | 0,7760 | 0,5936 | 0,6229 | 0,7841 |
| Francia | 1.792 | 15,96 % | 13,28 % | 86,72 % | 0,7647 | 0,6364 | 0,6585 | 0,7910 |
| Alemania | 542 | 41,88 % | 38,38 % | 61,62 % | 0,8029 | 0,7357 | 0,7482 | 0,8865 |
| España | 666 | 15,92 % | 12,61 % | 87,39 % | 0,8095 | 0,6415 | 0,6693 | 0,8003 |

Las tasas de decisión son las proporciones clasificadas como `contactar` o
`sin_contacto` al umbral 0,50. El F2 y la average precision muestran una brecha
marcada por edad: el recall es 0,1942 para menores de 40 y 0,8125 para 40 o
más. También hay diferencias por género y geografía. Las tasas de churn
observadas varían entre grupos, por lo que las tasas de contacto no bastan para
concluir que exista o no trato injusto. La aceptabilidad para clientes y negocio
no se puede establecer sin una política bancaria validada, resultados de una
intervención y costos de error. Hasta entonces, no se automatiza el contacto.

## Costos de error y supervisión

- **Falso positivo:** una persona que no habría cancelado recibe una revisión o
  un contacto innecesario; el banco incurre en costo operativo y puede afectar
  la experiencia, confianza o privacidad de la persona.
- **Falso negativo:** una persona que cancela no recibe una oferta o asistencia
  que podría haber sido útil; el banco puede perder la relación y la persona,
  una opción pertinente. No hay datos para asignar valores monetarios a ninguno
  de estos efectos.
- En modo normal, `contactar` genera una recomendación de
  `revision_humana_retencion`, no una llamada automática. La persona responsable
  debe revisar pertinencia, consentimiento y política antes de actuar. El
  proyecto todavía no registra la decisión humana ni el resultado de la acción.
- El juicio humano de PSet 1 se formuló para telecomunicaciones. No se traslada
  como regla a clientes bancarios; la revisión humana aquí es una salvaguarda
  propuesta, pendiente de validación por el área responsable.

## Criterios de respaldo o retirada

| Indicador | Umbral operativo | Respuesta |
|---|---|---|
| Estado del servicio o fallback | `/modelo` informa `degraded` o la proporción de predicciones con `USED_FALLBACK=true` es mayor que 0 en cualquier lote. | Suspender decisiones automáticas, activar `manual_review_all` y revisar el modelo, alias y dependencia que falló. |
| Deriva de una feature | PSI supera `0.25` en cualquier feature de un lote. Es el umbral alto configurado en `monitoreo/alertas.py`. | Emitir alerta, revisar calidad y población; derivar casos a revisión humana. La deriva sola no autoriza reentrenar. |
| Rendimiento con outcomes | En dos cohortes consecutivas ya etiquetadas, F2 del champion es menor que F2 de la regresión logística evaluada en las mismas cohortes. | Mantener revisión humana, investigar y no promover un candidato hasta una comparación independiente y aprobación humana. |

El tercer criterio no se puede evaluar con el test de competencia actual porque
no tiene `exited`; tampoco se debe reciclar el holdout consumido como si fueran
cohortes futuras.

## Interruptor de revisión manual

Configurar `PREDICTION_MODE=manual_review_all` en `.env` y recrear solo la API:

```bash
docker compose up -d --force-recreate api
```

En ese modo la API no consulta MLflow ni usa el modelo para decidir; cada
respuesta lleva `decision="revision_humana"`, `accion="revision_humana_retencion"`
y `used_fallback=true`. `probabilidad` conserva el prior global del fallback
(`0.2064167`), no es un score individual ni debe ordenarse como riesgo. Para
volver al modo del modelo, configurar `PREDICTION_MODE=model` y recrear la API.
El fallback actual no es la regla comercial de PSet 1: esa regla de telecom no
se puede calcular con las variables bancarias disponibles.
