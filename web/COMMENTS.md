# Configurar Giscus (3 pasos · ~2 minutos)

La página tiene una caja de comentarios al final, justo antes del footer.
Está soportada por [Giscus](https://giscus.app), que guarda los comentarios
como GitHub Discussions en tu propio repo. Comentar requiere cuenta de
GitHub (eso filtra spam de paso).

Para que funcione hay que activar tres cosas en el repo:

## 1. Activar Discussions

Repo → **Settings** → **General** → sección "Features" →
marca **Discussions**.

## 2. Instalar la app de Giscus

Ve a <https://github.com/apps/giscus> → **Install** →
selecciona `BioscaG/transformer-structural-compression-v2` (no hace falta
darle acceso a todo, sólo a este repo).

## 3. Crear la categoría de comentarios y meter el ID

a. Repo → pestaña **Discussions** → **New discussion** → **New
   category** → llámala `Comments` con tipo **Announcement** (sólo
   maintainers pueden abrir hilos nuevos; el resto sólo responde).

b. Ve a <https://giscus.app>:
   - Repo: `BioscaG/transformer-structural-compression-v2`
   - Mapeo: **pathname**
   - Categoría: **Comments**

   El sitio te genera un snippet `<script>`. Dentro hay
   `data-category-id="DIC_kw..."`. Copia ese valor.

c. Abre `web/sections.py` y reemplaza `REPLACE_WITH_CATEGORY_ID` por
   el `DIC_kw...` que copiaste:

   ```python
   COMMENTS = {
       ...
       "giscus": {
           ...
           "category_id":  "DIC_kw...",   # ← aquí
           ...
       },
   }
   ```

d. Reconstruye el sitio y push:

   ```bash
   .viz_venv/bin/python -m web.build_index
   git add -A && git commit -m "comments: enable giscus" && git push
   ```

Listo. Visita `anatomy.guidobiosca.com#comentarios` y deberías ver
el widget cargando.

## Notas

- **Idioma**: la caja se carga en `es` o `en` según el toggle del sitio,
  y se recarga si cambias de idioma a media página.
- **Mapeo `pathname`**: un único hilo por página. Si quieres hilos
  separados por figura, cambia `mapping` a `specific` y crea hilos
  manualmente con título = id de la sección. Más trabajo de mantener.
- **Modo presentación**: la caja se oculta cuando entras en `?present=1`
  para que la defensa no se vea contaminada por comentarios.
- **Si no quieres comentarios**: borra `render_comments()` de
  `web/build_index.py` (la línea de `body = ... + render_comments() +
  render_footer()`) y rebuild.
