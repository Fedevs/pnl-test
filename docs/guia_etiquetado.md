# Guía de etiquetado

Esta guía define cómo se asigna la categoría *gold* a cada conversación. La usan el generador (en el prompt), los revisores humanos y los prompts de los clasificadores LLM. **Cada conversación tiene exactamente una etiqueta.**

## 1. Categorías

### `pagos` — Pagos y cobros
Problemas o dudas con el **dinero de una compra que no se va a devolver**: medios de pago, pago rechazado, cobro duplicado, cuotas e intereses, factura o comprobante, cobro no reconocido.
- ✅ "me cobraron dos veces la misma compra"
- ✅ "puedo pagar en 6 cuotas sin interes con naranja?"
- ✅ "me rechaza la tarjeta y tengo saldo"
- ⚠️ Si el cliente **pide que le devuelvan la plata** de un cobro duplicado → sigue siendo `pagos` (es un error de cobro, no la devolución de un producto). Si pide la plata porque **devuelve o canceló el producto** → `devoluciones`.

### `envios` — Envíos y entregas
Todo lo que pasa entre que se confirma la compra y el producto llega: plazos, seguimiento, demoras, dirección de entrega, costo de envío, retiro en sucursal, paquete marcado como entregado que no llegó.
- ✅ "hace 10 dias que dice en camino y nada"
- ✅ "puedo cambiar la direccion si ya lo despacharon?"
- ⚠️ Si el producto llegó **dañado o equivocado** y el cliente quiere devolverlo o cambiarlo → `devoluciones`. Si solo informa que llegó tarde → `envios`.

### `devoluciones` — Devoluciones y reembolsos
Devolver, cambiar o cancelar una compra, y el reembolso asociado: producto fallado, talle equivocado, arrepentimiento, estado de un reembolso, cancelación de pedido.
- ✅ "quiero devolver las zapatillas, me quedaron chicas"
- ✅ "cancelé el pedido hace una semana y no me devolvieron la plata"
- ✅ "me llegó roto, quiero que me manden otro"

### `cuenta` — Cuenta y acceso
Acceso y datos de la cuenta del usuario: login, contraseña, código de verificación, cuenta bloqueada o suspendida, cambio de email o teléfono, datos personales, seguridad (acceso no autorizado).
- ✅ "no me llega el codigo para entrar"
- ✅ "me hackearon la cuenta"
- ⚠️ Una compra no reconocida hecha desde la cuenta hackeada: si el foco es **recuperar la cuenta** → `cuenta`; si es **el cobro** → `pagos`. Si aparecen las dos cosas, aplica la prelación (§2): gana `pagos`.

### `productos` — Productos y publicaciones
Información sobre productos antes o fuera de una compra concreta: características, stock, talles, compatibilidad, garantía, precio publicado, preguntas sobre una publicación, problemas al publicar (vendedores).
- ✅ "esta notebook trae windows?"
- ✅ "cuando vuelve a haber stock del talle M?"
- ⚠️ Garantía de un producto **ya comprado que falla** y el cliente quiere cambiarlo o devolverlo → `devoluciones`. Si solo pregunta qué cubre la garantía → `productos`.

## 2. Regla de prelación (consultas multi-intención)

Si una conversación contiene **más de una intención accionable**, la etiqueta es la de mayor prioridad:

```
devoluciones > pagos > cuenta > envios > productos
```

**Justificación de negocio:** se prioriza lo que involucra dinero que el cliente reclama y tiene mayor riesgo de pérdida o de contracargo (devoluciones y reembolsos, luego pagos), después la seguridad y el acceso (cuenta) y, por último, lo logístico e informativo.

Ejemplos:
| Conversación | Intenciones | Etiqueta |
|---|---|---|
| "no me llegó el pedido y quiero que me devuelvan la plata" | envios, devoluciones | `devoluciones` |
| "me cobraron el envío dos veces y encima no llegó" | pagos, envios | `pagos` |
| "no puedo entrar a la cuenta para ver el seguimiento" | cuenta, envios | `cuenta` |
| "tiene garantía? y cuánto tarda en llegar a Córdoba?" | productos, envios | `envios` |

Se considera intención **accionable** algo que el cliente pide resolver o preguntar. Las menciones de contexto no cuentan ("lo compré con tarjeta y quiero cambiar el talle" → solo `devoluciones`).

En el dataset, el campo `intenciones_presentes` lista todas las intenciones y `categoria_gold` es la resultante de aplicar la prelación.

## 3. Niveles de dificultad (campo `dificultad`)

| Nivel | Descripción | Proporción objetivo |
|---|---|---|
| `facil` | Una intención clara, lenguaje estándar | ~50% |
| `ambigua` | Una intención, pero expresada de forma vaga o con vocabulario de otra categoría ("tengo un problema con mi compra") | ~15% |
| `disfluente` | Una intención con ruido: typos, sin tildes, abreviaturas ("q", "xq"), voseo, mayúsculas, audio transcrito, mensajes partidos | ~15% |
| `multi_intencion` | Dos o más intenciones; se resuelve con §2 | ~20% |

## 4. Fuera de dominio

Saludos sin consulta, spam, quejas genéricas sin motivo identificable o temas ajenos (empleo, prensa): **se descartan del dataset**. No existe una sexta clase "otros". Esto se declara como limitación en el informe.

## 5. Procedimiento de revisión humana

La categoría gold **nunca se escribe a mano**: se marcan las intenciones presentes y la categoría sale de aplicar la prelación (§2). Así la regla se aplica igual en todos los casos.

1. `python data/prepare.py review-sheets --members <nombres>` genera dos planillas por integrante en `data/review/`. Están separadas por `;` y se abren directo en Excel.
2. **`<nombre>_revision.csv`** tiene la porción propia. Por cada fila:
   - Si está bien, se deja `accion = ok`.
   - Si es inverosímil o está fuera de dominio, se pone `accion = descartar`.
   - Si las intenciones están mal, se completa `intenciones_corregidas` con códigos separados por `|`, por ejemplo `pagos|envios`.
   - Si la dificultad está mal, se completa `dificultad_corregida`.
3. **`<nombre>_ciego.csv`** tiene un 12% compartido entre todos y **sin la etiqueta del generador**. Cada integrante completa `intenciones` (y `descartar` con una `x` si corresponde) sin mirar a los demás. Con esto se calcula el **kappa de Fleiss**.
4. `python data/prepare.py apply-review` aplica las correcciones. En el subconjunto ciego se queda la mayoría y los empates van a `data/review/desacuerdos.csv`. El script reporta el kappa y cuántas etiquetas del generador se corrigieron (indicador de calidad del generador) en `results/revision_humana.json`.
5. Los desacuerdos se discuten. Si surge un criterio nuevo, se agrega a esta guía con un ejemplo.
