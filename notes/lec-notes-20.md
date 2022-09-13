[toc]

<hr style="border: 5px solid #003262;" />
<hr style="border: 1px solid #fdb515;" />

## Lecture 18, 03/29/22 (Wk10): SQL 1

### Associated Lecture Resources

- [slides, code, code HTML, recording](https://ds100.org/sp22/lecture/lec18/)

### Brief Databases Overview

A **database management system (DBMS)** is a software system that stores, manages, and facilitates access to one or more **databases**, an organizsed collection of data.

In the real world, data might not be in CSV files. Storing data in a DBMS is useful for non-CSV data types, but we must use languages such as **SQL** (Structured Query Language) to query for our data.

**Data Storage:**

- Reliable storage to survive system crashes and disk failures
- Optimize to compute on data that does not fit in memory
- Special data structures to improve performance

**Data Management:**

- Configure how data is organized and who has access
- Enforce guarantees on the data (e.g. non-negative bank account balance)
  - Can be used to prevent data anomolies
  - Ensures safe concurrent operations on data

### Summary of SQL Syntax from Today

```SQL
SELECT <column list>
FROM <table>
[WHERE <predicate>]
[ORDER BY <column list>]
[LIMIT <number of rows>]
[OFFSET <number of rows>]
```

```SQL
SELECT [DISTINCT] <column expression list>
FROM <table>
[WHERE <predicate>]
[GROUP BY <column list>]
[HAVING <predicate>]
[ORDER BY <column list>]
[LIMIT <number of rows>]
[OFFSET <number of rows>]
```

Note: Column expressions may include the aggregation functions `MAX`, `MIN`, and/or `DISTINCT`.

### The Magic Comamnd

iPython knows we're writing in SQL when we write the following cell at the top of a cell: `%%sql`

### SQL Terminology

- use singular, [**CamelCase** ](https://en.wikipedia.org/wiki/Camel_case) names for SQL tables
- **rows** are called **records** or **tuples**
- **columns** are called **attributes** or **fields**
- **tables** are called **relations**

<img src="lec-notes.assets/image-20220901205713064.png" alt="image-20220901205713064" style="zoom:25%;" />

- 主键 primary key (PK)

<img src="lec-notes.assets/image-20220901205854239.png" alt="image-20220901205854239" style="zoom:25%;" />



#### Properties

**Every field in a table has three properties:**
- ColName
- Type
- Constraints

#### Types

- **INT:** Integers
- **REAL:** Real numbers $\mathbb{R}$
- **TEXT:** Strings of text
- **BLOB:** Arbitrary data, e.g. songs, video files, etc.
- **DATETIME:** A date and time

<img src="lec-notes.assets/image-20220901210146619.png" alt="image-20220901210146619" style="zoom:25%;" />

#### Constraints

- **CHECK:** Data cannot be inserted
- **PRIMARY KEY:** Specifies that this key is used to uniquely identify rows in the table
- **NOT NULL:** Null data cannot be inserted for this column
- **DEFAULT:** Provides a value to use if the user does not specify this on insertion
ss

#### Primary Keys

<img src="lec-notes.assets/image-20220901210619123.png" alt="image-20220901210619123" style="zoom:25%;" />

### SQL Keywords

- `SELECT`

    ```SQL
    %%sql
    SELECT * FROM Table; -- select all rows and columns
    
    SELECT col1, col2 FROM Table; -- select a subset of the columns
    ```

- `AS`
    ```SQL
    %%sql
    SELECT cute as cuteness FROM Table; -- change the column name
    ```

    More readable:

    ```sql
    %%sql
    SELECT cute AS cuteness,
    	   year AS birth
    FROM Dragon
    ```

- `WHERE`

    ```SQL
    %%sql
    SELECT col1, col2 FROM Table WHERE cute > 0; -- select only some rows of a table based on a condition
    
    SELECT col1, col2 FROM Table WHERE cute > 0 OR year > 2021;
    ```

    More readable:

    ```sql
    SELECT name, cute, year
    FROM Dragon
    WHERE cute > 0 OR year > 2013;
    ```

- `OR`

    ```SQL
    %%sql
    SELECT col1, col2 FROM Table WHERE cute > 0 OR year > 2021;
    ```

- `ORDER BY`

    ```SQL
    %%sql
    SELECT * FROM Table ORDER BY cute DESC; -- select all columns of Table and order by cute in descending order
    ```

    - `ASC` 为升序
    - 每次执行 `SELECT * FROM Dragon`, 各列的顺序都是随机的（

- `LIMIT`

    ```SQL
    %%sql
    SELECT * FROM Table LIMIT 2; -- limit the number of result rows to 2
    ```

- `OFFSET`

    Note that, unless you use `ORDER BY`, there is no guaranteed order!

    ```SQL
    %%sql
    SELECT * From Table LIMIT 2 OFFSET 10; -- see 10 later rows when limiting
    ```

- `GROUP BY`

    <img src="lec-notes.assets/image-20220901212615404.png" alt="image-20220901212615404" style="zoom:25%;" />

    <img src="lec-notes.assets/image-20220901212913096.png" alt="image-20220901212913096" style="zoom:25%;" />

    - 类似地还有 `MAX()`, `MIN()`, `COUNT(*)`, ...

        > count 一个特定的列名，其计算的非空记录的数量，count * 则计算列总数

    <img src="lec-notes.assets/image-20220901213127873.png" alt="image-20220901213127873" style="zoom:25%;" />

    <img src="lec-notes.assets/image-20220901214031124.png" alt="image-20220901214031124" style="zoom:25%;" />

    同时 group by 两列：

    <img src="lec-notes.assets/image-20220901214142516.png" alt="image-20220901214142516" style="zoom:25%;" />

    <img src="lec-notes.assets/image-20220901214547640.png" alt="image-20220901214547640" style="zoom:25%;" />

- `HAVING`

    ```SQL
    %%sql
    SELECT col1, COUNT(*)
    FROM Table
    GROUP BY col1
    HAVING MAX(col2) < 5;
    ```
    
    This above query is equivalent to the following pandas code:

    ```py
    df.groupby('col1').filter(lambda f: max(f['col2'] < 5))
    ```

    <img src="lec-notes.assets/image-20220901214808369.png" alt="image-20220901214808369" style="zoom:25%;" />

    <img src="lec-notes.assets/image-20220901214933164.png" alt="image-20220901214933164" style="zoom:25%;" />

    - To filter rows, use `WHERE`
    - To filter groups, use `HAVING`
    - `WHERE` happens before `HAVING`


- `DISTINCT`

    <img src="lec-notes.assets/image-20220901220049903.png" alt="image-20220901220049903" style="zoom:25%;" />

    <img src="lec-notes.assets/image-20220901220220761.png" alt="image-20220901220220761" style="zoom:25%;" />

```SQL
%%sql
SELECT col1, MAX(col2) FROM Table GROUP BY col1; -- select the maximum value from col2 for each unique value in col1
```

---

<br><br>

## Lecture 19, 03/31/22 (Wk10): SQL 2

### Downsides of `pd.read_sql()`

1. Using code as an argument to `pd.read_sql()` is error prone, but the error messages are cryptic: 

Instead, we'll often write SQL code using the SQL Magic command

<img src="lec-notes.assets/image-20220902105155527.png" alt="image-20220902105155527" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220902105212227.png" alt="image-20220902105212227" style="zoom:25%;" />

### .tsv Files

.tsv files contain tab separated data, but we need to convert .tsv files into .db format so that SQL can be used instead. There is a command on the DataHub machines called sqlite3, which compresses the file and puts it in a format so that we can use sqlalchemy to query the data

Let's create a sqlalchemy connection:

- :
    ```python
    engine = sqlalchemy.create_engine("sqlite:///data/database.db")
    ```

Now, we can request the list of Tables:
```py
tables = pd.read_sql("SELECT sql FROM sqlite_master WHERE table='table';", connection)
table1 = tables["sql"][0]
table2 = tables["sql"][1]
...
```

If we tried `pd.read_csv()` on the data before we did the above, we'd get an error; `pd_read_csv()` can't handle a ton of data in a file of size, say, 700 MB

### `LIKE`

The `LIKE` operator tests whether a string matches a pattern (similar to a regex, but with much simpler syntax). To select movies from a table `t` that has a column `time` where the string is on the hour, we would write the following:

```SQL
%%sql
SELECT * FROM t WHERE t.time LIKE ‘%:00%’
```

<img src="lec-notes.assets/image-20220902110030009.png" alt="image-20220902110030009" style="zoom:25%;" />



### `CAST`

The `CAST` keyword converts a table column to another type. This keyword is particularly useful if we're missing data

<img src="lec-notes.assets/image-20220902110402972.png" alt="image-20220902110402972" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220902110417423.png" alt="image-20220902110417423" style="zoom:25%;" />

### Joins

Example of an **inner join**:

<img src="lec-notes.assets/image-20220902112435463.png" alt="image-20220902112435463" style="zoom:25%;" />

A **cross join** puts all pairs of rows together in the resulting table. This is also known as the **Cartesian product**:

<img src="lec-notes.assets/image-20220902112649706.png" alt="image-20220902112649706" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220902112824978.png" alt="image-20220902112824978" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220902113017629.png" alt="image-20220902113017629" style="zoom:25%;" />

For more examples and use of the `JOIN` keyword, check out [slide 27 and beyond](https://docs.google.com/presentation/d/1EdxE8dlOpaJ09aloqeR9-3avz4f5fVTeZgfWSgbL6DM/edit#slide=id.g120e623da8f_1_209)

<img src="lec-notes.assets/image-20220902113123761.png" alt="image-20220902113123761" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220902113355519.png" alt="image-20220902113355519" style="zoom:25%;" />

### PCA

**PCA** is a technique for high dimensional EDA and featurization.

If we have a dataset of **N** observations with **d** attributes, we can think of the data as being **N** vectors in a **d**-dimensional space. For example, a dataset with 3 columns will be in 3-dimensional space.

Matrix decomposition (a.k.a Matrix Factorization) is the opposite of matrix multiplication: taking a matrix and decomposing it into two separate matrices.

**Rank** is the maximum number of linearly independent rows in a Table (matrix)

### Matrix Decomposition Summary

Decomposing an MxN matrix `X` into two matrix factors of size MxQ and QxN does not yield a unique answer.

- The size of ==the smallest matrices are when Q = the rank R, i.e. MxR and RxN==

<img src="lec-notes.assets/image-20220902121102655.png" alt="image-20220902121102655" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220902121122329.png" alt="image-20220902121122329" style="zoom:25%;" />

**Singular Value Decomposition** decomposes `X` into matrices of size MxN and NxN, where

- The left matrix is no longer the oroginal matrix
- The right atrix transforms the data
- If rank < N, then the N - rank columns will all be zeros (so we can get rid of them); this one might not be intuitive, but worry not–this will probably show up in the homework 

---

<br><br>

## Lecture 20, 04/05/22 (Wk11): PCA II

### Principal Component Analysis

**Principal Component Analysis (PCA)** is the process of linearly transforming data into a new coordinate system such that the greatest variance occurs along the first dimension, the second most along the second dimension

**PCA is appropriate for EDA when:**
- Visually identifying clusters of similar observations in high dimensions (greater than R2)
- You are still exploring the data
- You have reason to believe that the data are inherently low rank: there are many attributes, but only a few determine the rest through a linear association


### Singular Value Decomposition

**Singular Value Decomposition** automatically decomposes a matrix into two matrices
- left matrix U$\mathcal{E}$ has same dimensionality as original data X
- right matrix $V^T$ will transform U$\mathcal{E}$ back into X
- $\mathcal{E}$ is a diagonal matrix; contains the "singular values" of X
- columns of U and V are an orthonormal set

<img src="lec-notes.assets/image-20220903195822018.png" alt="image-20220903195822018" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220903200321082.png" alt="image-20220903200321082" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220903200400754.png" alt="image-20220903200400754" style="zoom:25%;" />

- 可以 drop 掉最后一行、最后一列

<img src="lec-notes.assets/image-20220903200627694.png" alt="image-20220903200627694" style="zoom:25%;" />

- 近似地，可以 drop 掉最后两行、最后两列（甚至三）

**Orthonormal vectors:**

- are all unit vectors, i.e. have length 1
- vectors are orthogonal

<img src="lec-notes.assets/image-20220903204032502.png" alt="image-20220903204032502" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220903204340368.png" alt="image-20220903204340368" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220903204443318.png" alt="image-20220903204443318" style="zoom:25%;" />

### Principal Component

When performing SVD, we've left the columns in the "data" matrix U$\mathcal{E}$ unnamed;
- Their common name: "Principal Components"
- Center data first to get the correct principal components

Compute the fraction of the variance captured in each coordinate:
```py
np.round(s**2 / sum(s**2), 2)
plt.plot(s**2)
```

### Summary

<img src="images/../../images/lec-20-0.png" style="zoom: 50%;" >

<img src="lec-notes.assets/image-20220903205933991.png" alt="image-20220903205933991" style="zoom:25%;" />

<img src="lec-notes.assets/image-20220903210244789.png" alt="image-20220903210244789" style="zoom:25%;" />

### SVD vs Linear Regression

<img src="lec-notes.assets/image-20220903212105773.png" alt="image-20220903212105773" style="zoom:25%;" />

---

<br><br>
