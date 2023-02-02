# Some Benchmarks

## Encoding benchmarks

### Pack/Unpack Function (1 000 000 Iterations)

```
+---------------+----------------+------------+
|    Function   | Implementation | Time (sec) |
+---------------+----------------+------------+
|      pack     |     Python     |    0.58    |
|     unpack    |     Python     |    0.52    |
|      pack     |      C++       |    0.37    |
|     unpack    |      C++       |    0.35    |
|  pack (fast)  |      C++       |    0.36    |
| unpack (fast) |      C++       |    0.22    |
+---------------+----------------+------------+
```
### Pack Tensor Function (1 000 000 Iterations)

```
+-------------+----------------+------------+
|   Function  | Implementation | Time (sec) |
+-------------+----------------+------------+
| pack_tensor |     Python     |    27.4    |
| pack_tensor |      C++       |    2.86    |
+-------------+----------------+------------+
```

### Unpack Tensor Function (1 0000 000 Iterations)

Note: Higher numbers more diverges

```
+---------------+----------------+------------+
|    Function   | Implementation | Time (sec) |
+---------------+----------------+------------+
| unpack_tensor |     Python     |    8.21    |
| unpack_tensor |      C++       |    2.05    |
+---------------+----------------+------------+

```
### n_hot unpacked Function (1 000 000 Iterations)

```
+----------+----------------+------------+
| Function | Implementation | Time (sec) |
+----------+----------------+------------+
|  n_hot   |     Python     |    2.65    |
|  n_hot   |      C++       |    1.8     |
+----------+----------------+------------+
```

### n_hot packed function (1 000 000 Iterations)

```
+----------+----------------+------------+
| Function | Implementation | Time (sec) |
+----------+----------------+------------+
|  n_hot   |     Python     |   22.15    |
|  n_hot   |      C++       |    5.92    |
+----------+----------------+------------+
```

### Embedding Layer function (1 000 000 Iterations)

```
+--------------------+----------------+------------+
|      Function      | Implementation | Time (sec) |
+--------------------+----------------+------------+
| Embedding Unpacked |     Python     |   44.66    |
|  Embedding Packed  |     Python     |   520.91   |
| Embedding Unpacked |      C++       |   43.62    |
|  Embedding Packed  |      C++       |   42.72    |
+--------------------+----------------+------------+
```



### Embedding Layer (1 Iteration)

```
| Type | implementation | Device  | Time  |
|---|---|---|---|---|
| Packed  | Python | cpu | 1.51ms  |
| Packed  | Python | cuda | 3.59ms  | 
| Unpacked  | Python | cpu | 127.35 us  |
| Unpacked  | Python | cuda | 167.61 us  |
```




