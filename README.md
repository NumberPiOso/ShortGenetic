# SHORT GENETIC

Enmarcado como parte del proyecto **MOCO** (MultiObjective Combinatorial Optimization) en el cual se busca un algoritmo genético más simple para el problema **BRP**.

Esta organizado de la siguiente manera:

<!-- **code**: Código y resultados   -->
**utilities.py**: El código que contiene funciones de apoyo para model.py, funciones que no son en sí del genético pero que       pueden hacer la programación más fácil, como bloquear prints a consola o encontrar ciertas métricas entre fronteras.  
**tunning_params/**: En esta carpeta se guardaran los códigos y resultados del tuneo de parametros.  
**test_sols.py**: Código para debuggear el problema y mostrar en consola algunos resultados parciales de correr el genético.  
**model.py**: En este código se encuentran los objetos **Sol**, **SolCollection** y **Station**, que son respectivamente, una solución, la colección de soluciones y estaciones.  
**main.py**: Lugar donde hago los ejemplos de como se haría el genético.  
**exact_models_cpp**: Es donde corremos diferentes tipos de modelos exactos para posteriormente ser comparados con las soluciones del genético.  
**cython_build**: Algunas construcciones de los méthodos más lentos en Python y C, para hacerlos más rápidos.
**Avances**: En esta carpeta se guardaran los avances que se vayan haciendo del problema, del tal manera que se pueda mantener una muestra de que se ha ido haciendo.

### Ultimos avances

- Túneo de parámetros.  
- Modelos exactos c++.  
[Verlos acá](./code/avances/0410/avances_0410.ipynb) 