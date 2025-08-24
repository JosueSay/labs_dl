# Documentación de capas CNN en PyTorch

## 1. `nn.Conv2d`

**Descripción:**

Aplica una **convolución 2D** sobre una entrada con múltiples canales. Es la capa base de las CNN para extraer rasgos espaciales y locales.

**Entrada esperada:**

- Tensor de forma `(N, C_in, H, W)`

  - `N`: tamaño del batch
  - `C_in`: canales de entrada (ej. 1 para MNIST, 3 para RGB)
  - `H, W`: alto y ancho de la imagen

**Salida:**

- Tensor de forma `(N, C_out, H_out, W_out)`

**Parámetros principales:**

- `in_channels (int)` -> número de canales de entrada.
- `out_channels (int)` -> número de filtros (canales de salida).
- `kernel_size (int o tuple)` -> tamaño del filtro (ej. 3 o (3,3)).
- `stride (int o tuple, opcional)` -> desplazamiento del kernel. Default: 1.
- `padding (int, tuple o str)` -> relleno en los bordes. Default: 0.

  - `'valid'` = sin padding.
  - `'same'` = salida mantiene tamaño de entrada.
- `dilation (int o tuple)` -> espaciado entre elementos del kernel. Default: 1.
- `groups (int)` -> conexiones entrada/salida.

  - `1`: todos los canales conectados.
  - `in_channels`: depthwise conv (cada canal con su propio filtro).
- `bias (bool)` -> si añade sesgo aprendible. Default: True.
- `padding_mode (str)` -> `'zeros'`, `'reflect'`, `'replicate'`, `'circular'`.

**Uso típico en CNN:**

- Primeras capas convolucionales con `kernel_size=3` o `5`.
- `stride=1` para no perder información.
- `padding="same"` para mantener resolución.
- `groups=in_channels` en convoluciones separables (arquitecturas móviles).

## 2. `nn.MaxPool2d` / `torch.nn.functional.max_pool2d`

**Descripción**
Aplica una operación de max-pooling 2D: selecciona el valor máximo dentro de una ventana, reduciendo resolución espacial y aportando invariancia a traslaciones locales.

**Entrada esperada:**

- Tensor `(N, C, H_in, W_in)`

**Salida:**

- Tensor `(N, C, H_out, W_out)`

**Parámetros principales:**

- `kernel_size (int o tuple)` -> tamaño de la ventana.
- `stride (int o tuple)` -> paso de la ventana. Default: `kernel_size`.
- `padding (int o tuple)` -> relleno implícito con `-inf`. Default: 0.
- `dilation (int o tuple)` -> controla espaciamiento entre puntos de la ventana.
- `return_indices (bool)` -> devuelve también índices del máximo (útil en `MaxUnpool2d`).
- `ceil_mode (bool)` -> si se usa `ceil` en lugar de `floor` al calcular salida.

**Uso típico en CNN:**

- Reducción espacial con `kernel_size=2`, `stride=2`.
- Bloques conv -> ReLU -> MaxPool.
- Control del downsampling progresivo de la imagen.

## 3. `nn.AvgPool2d` / `torch.nn.functional.avg_pool2d`

**Descripción**
Aplica un promedio sobre regiones locales en lugar del máximo. Reduce la resolución suavizando la representación.

**Entrada esperada:**

- Tensor `(N, C, H_in, W_in)`

**Salida:**

- Tensor `(N, C, H_out, W_out)`

**Parámetros principales:**

- `kernel_size (int o tuple)` -> tamaño de la ventana.
- `stride (int o tuple)` -> paso de la ventana. Default: `kernel_size`.
- `padding (int o tuple)` -> relleno con ceros.
- `ceil_mode (bool)` -> si se usa `ceil` en la fórmula de salida.
- `count_include_pad (bool)` -> si el padding cuenta en el promedio. Default: True.
- `divisor_override (int)` -> divisor manual para el cálculo del promedio.

**Uso típico en CNN:**

- Sustituto de max-pooling cuando interesa suavizar en vez de filtrar máximos.
- Global Average Pooling (GAP): usar `kernel_size=(H_in, W_in)` para obtener un vector por canal antes de la capa densa final (ej. ResNet, EfficientNet).

## 4. Normalización del dataset MNIST

**Problema:**
MNIST tiene valores de píxel en `[0,255]`. Si no se normaliza, la red puede aprender más lento y tender a sobreajustar.

**Pasos comunes de normalización:**

1. Escalado a \[0,1]

    ```python
    x = x / 255.0
    ```

2. Estandarización por canal

    Calcular media y desviación en el set de entrenamiento y transformar:

    $$
    x' = \frac{x - \mu}{\sigma}
    $$

    Para MNIST (un solo canal), se usan valores aproximados:

    - `mean ≈ 0.1307`
    - `std ≈ 0.3081`

    ```python
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ```

3. Regularización adicional

    - `BatchNorm2d` en capas intermedias para estabilizar activaciones.
    - `Dropout` o `weight_decay` para mitigar sobreajuste.
