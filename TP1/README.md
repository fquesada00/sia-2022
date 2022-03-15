# TP1 SIA

Este trabajo presenta la implementación de distintos métodos de búsqueda (informados y no informados) para la resolución del **_"8 number puzzle"_**

![problem image](https://media.cheggcdn.com/study/bf2/bf22136e-2fb3-4bd0-8220-235c43b9c002/image.png)

## Instalación

Utilizar el manejador de paquetes [pip](https://pip.pypa.io/en/stable/) para instalar pipenv.

```bash
pip install pipenv
```

Luego instalar las dependencias del entorno.

```bash
pipenv install
```

## Uso

Inicializar el ambiente.

```bash
pipenv shell
```

Editar el archivo [config.json](config.json) y establecer los parámetros deseados para la búsqueda.

Ejecutar el archivo [main.py](main.py)

```bash
python main.py [-f, --file archivo de configuracion]
```

Argumentos opcionales:

```
-f, --file    Nombre del archivo de configuracion. Por defecto es config.json
```

## Configuración

El archivo de configuración es un json con el siguiente formato:

```json
{

	"grid_size": number,
	"initial_state":number[],
	"search_method": "DFS" | "BFS" | "DLS" | "IDS" | "LHS" | "GHS" | "A_STAR",
	"heuristic": "manhattan" | "euclidean" | "misplaced_tiles" | "misplaced_tiles_value" | "visited_tiles_value",
	"iterative_depth_search_initial_limit": number | null,
	"depth_limited_search_max_depth": number | null,
	"generate_visualizations": boolean,

}
```

Cuyas propiedades son:

- grid_size: Tamaño de un lado de la grilla
- initial_state: el estado inicial de la grilla (donde el espacio vacio se denota con un 0).
- search_method: Método con el se realizará la busqueda de la solución. Debe ser un método valido.
- heuristic: Heurística a utilizar en caso de utilizar un método de búsqueda informado (mayúsculas y minúsculas indistinguibles).
- iterative_depth_search_initial_limit: Limite inicial para la búsqueda de la solución utilizando Iterative Depth Search (IDS).
- depth_limited_search_max_depth: Profundidad máxima para la búsqueda de la solución utilizando Depth Limited Search (DLS).
- generate_visualizations: Indica si se generan las gráficas de las soluciones, siendo estas `tree.html` (gráfico del árbol de búsuqeda) y `steps.html` (solución paso a paso con tablero).

## Proximos pasos

- [x] Implementacion de métodos de búsqueda informados y no informados
- [ ] Permitir cualquier estado objetivo
