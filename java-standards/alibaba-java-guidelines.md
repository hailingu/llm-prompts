# 阿里巴巴 Java 开发规范 - 核心要点

本文档提炼《阿里巴巴 Java 开发手册》的核心规范，供 java-coder-specialist agent 参考执行。

完整规范请参考：[阿里巴巴 p3c 仓库](https://github.com/alibaba/p3c)

## 一、命名规范

### 1.1 强制规范

- **类名**使用 UpperCamelCase 风格，DO/DTO/VO/DAO 等除外：`UserService`, `OrderDTO`
- **方法名、参数名、成员变量、局部变量**使用 lowerCamelCase：`getUserName()`, `orderList`
- **常量命名**全部大写，单词间用下划线隔开：`MAX_STOCK_COUNT`, `DEFAULT_CHARSET`
- **包名**统一使用小写，点分隔符之间有且仅有一个自然语义的英语单词：`com.alibaba.ai.util`

### 1.2 推荐规范

- 抽象类命名使用 Abstract 或 Base 开头
- 异常类命名使用 Exception 结尾
- 测试类命名以它要测试的类的名称开始，以 Test 结尾：`UserServiceTest`
- POJO 类中布尔类型变量都不要加 is 前缀，否则部分框架解析会引起序列化错误

## 二、常量定义

### 2.1 强制规范

- 不允许任何魔法值（即未经预先定义的常量）直接出现在代码中
- `long` 或 `Long` 赋值时，数值后使用大写 L：`2L`（不是 `2l`）
- 不要使用一个常量类维护所有常量，应按功能分类，分开维护

### 2.2 常量分类示例
```java
public class CacheKeyConstants {
    public static final String LOGIN_MEMBER_KEY = "login:member:key:";
}

public class ConfigConstants {
    public static final int MAX_RETRY_COUNT = 3;
    public static final long DEFAULT_TIMEOUT = 30000L;
}
```

## 三、代码格式

### 3.1 强制规范
- 大括号的使用约定：左大括号前不换行，左大括号后换行；右大括号前换行
- 左小括号和右边相邻字符之间不出现空格；右小括号和左边相邻字符之间也不出现空格
- `if/for/while/switch/do` 等保留字与左右括号之间都必须加空格
- 任何运算符左右必须加一个空格
- 缩进采用 4 个空格，禁止使用 tab 字符
- 单行字符数限制不超过 120 个，超出需要换行

### 3.2 换行规则
```java
// 正确示例
StringBuilder sb = new StringBuilder();
sb.append("规则一：")
  .append("换行时，运算符与下文一起换行")
  .append("规则二：")
  .append("换行时缩进 4 个空格");
```

## 四、OOP 规约

### 4.1 强制规范
- 避免通过一个类的对象引用访问此类的静态变量或静态方法，应直接用类名来访问
- 所有覆写方法，必须加 `@Override` 注解
- 相同参数类型，相同业务含义，才可以使用 Java 的可变参数
- 外部正在调用的接口或二方库依赖的接口，不允许修改方法签名，避免对接口调用方产生影响
- 不能使用过时的类或方法
- `Object` 的 `equals` 方法容易抛空指针异常，应使用常量或确定有值的对象来调用 equals

```java
// 推荐写法
"test".equals(object);
Objects.equals(obj1, obj2);  // JDK7 引入，可以避免空指针
```

### 4.2 POJO 类规约
- POJO 类属性必须使用包装数据类型（Integer, Long, Boolean 等）
- POJO 类中的任何布尔类型的变量，都不要加 is 前缀
- 序列化类新增属性时，不要修改 serialVersionUID，避免反序列失败

## 五、集合处理

### 5.1 强制规范
- 关于 `hashCode` 和 `equals` 的处理，遵循如下规则：
  - 只要覆写 equals，就必须覆写 hashCode
  - Set 存储的对象必须覆写这两个方法
- 使用集合的构造函数指定初始值大小：`new ArrayList<>(initialCapacity)`
- 使用 `entrySet` 遍历 Map 类集合，而不是 `keySet`

```java
// 推荐写法
Map<String, String> map = new HashMap<>(16);
for (Map.Entry<String, String> entry : map.entrySet()) {
    String key = entry.getKey();
    String value = entry.getValue();
}
```

### 5.2 推荐规范
- 集合初始化时，指定集合初始值大小
- 使用 `isEmpty()` 方法判断集合是否为空，而不是使用 `size() == 0`

## 六、并发处理

### 6.1 强制规范
- 获取单例对象需要保证线程安全，其中的方法也要保证线程安全
- 创建线程或线程池时请指定有意义的线程名称，方便出错时回溯
- 线程资源必须通过线程池提供，不允许在应用中自行显式创建线程
- 线程池不允许使用 `Executors` 去创建，而是通过 `ThreadPoolExecutor` 的方式

```java
// 推荐写法
ThreadPoolExecutor executor = new ThreadPoolExecutor(
    5, 10, 60L, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(100),
    new ThreadFactoryBuilder().setNameFormat("XX-task-%d").build(),
    new ThreadPoolExecutor.AbortPolicy()
);
```

### 6.2 锁规约
- 高并发时，同步调用应该考虑锁的性能损耗，能用无锁数据结构，就不要用锁
- 对多个资源、数据库表、对象同时加锁时，需要保持一致的加锁顺序，否则可能会造成死锁

## 七、控制语句

### 7.1 强制规范
- 在一个 switch 块内，每个 case 要么通过 break/return 等来终止，要么注释说明程序将继续执行到哪一个 case
- 在 if/else/for/while/do 语句中必须使用大括号
- 在高并发场景中，避免使用"等于"判断作为中断或退出的条件
- 表达异常的分支时，少用 if-else 方式

```java
// 推荐：卫语句
public void method(Param param) {
    if (param == null) {
        return;
    }
    // 正常业务逻辑
}
```

### 7.2 推荐规范
- 推荐使用三目运算符代替简单的 if-else 语句
- 循环体中的语句要考量性能，定义对象、变量、获取数据库连接等操作尽量移至循环体外处理

## 八、注释规约

### 8.1 强制规范
- 类、类属性、类方法的注释必须使用 Javadoc 规范
- 所有的抽象方法（包括接口中的方法）必须要用 Javadoc 注释
- 所有的类都必须添加创建者和创建日期
- 方法内部单行注释，在被注释语句上方另起一行，使用 `//` 注释

```java
/**
 * 用户服务实现类
 *
 * @author zhangsan
 * @date 2026-01-24
 */
public class UserServiceImpl implements UserService {
    
    /**
     * 根据用户ID查询用户信息
     *
     * @param userId 用户ID
     * @return 用户信息，不存在返回 null
     */
    @Override
    public User getUserById(Long userId) {
        // 参数校验
        if (userId == null || userId <= 0) {
            return null;
        }
        return userMapper.selectById(userId);
    }
}
```

### 8.2 推荐规范
- 特殊注释标记，标记人与标记时间：`TODO`, `FIXME`
- 代码修改的同时，注释也要进行相应的修改

## 九、异常处理

### 9.1 强制规范
- Java 类库中定义的可以通过预检查方式规避的 RuntimeException 异常不应该通过 catch 的方式来处理
- 异常不要用来做流程控制，条件控制
- catch 时请分清稳定代码和非稳定代码，稳定代码指的是无论如何不会出错的代码
- 捕获异常是为了处理它，不要捕获了却什么都不处理而抛弃之

```java
// 反例：生吞异常
try {
    method();
} catch (Exception e) {
    // 什么都不做
}

// 正例：至少记录日志
try {
    method();
} catch (Exception e) {
    log.error("Method execution failed", e);
}
```

### 9.2 事务规约
- 有 try 块放到了事务代码中，catch 异常后，如果需要回滚事务，一定要手动回滚事务
- 对于明确知道会产生唯一键冲突或大量数据库操作的情况，不要用事务

## 十、日志规约

### 10.1 强制规范
- 应用中不可直接使用日志系统（Log4j、Logback）中的 API，而应依赖使用日志框架 SLF4J
- 日志文件推荐至少保存 15 天
- 应用中的扩展日志（如打点、临时监控、访问日志等）命名方式：`appName_logType_logName.log`

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UserService {
    private static final Logger log = LoggerFactory.getLogger(UserService.class);
    
    public void method() {
        log.info("Processing user request");
        log.error("Error occurred", exception);
    }
}
```

### 10.2 推荐规范
- 谨慎地记录日志，生产环境禁止输出 debug 日志
- 可以使用 warn 日志级别来记录用户输入参数错误的情况

## 十一、MySQL 数据库

### 11.1 建表规约
- 表达是与否概念的字段，必须使用 is_xxx 的方式命名，数据类型是 unsigned tinyint（1 表示是，0 表示否）
- 表名、字段名必须使用小写字母或数字，禁止出现数字开头，禁止两个下划线中间只出现数字
- 主键索引名为 pk_字段名；唯一索引名为 uk_字段名；普通索引名则为 idx_字段名
- 小数类型为 decimal，禁止使用 float 和 double
- 表必备三字段：`id`, `create_time`, `update_time`

### 11.2 SQL 语句规约
- 不要使用 count(列名) 或 count(常量) 来替代 count(*)
- 当某一列的值全是 NULL 时，count(col) 的返回结果为 0
- 不得使用外键与级联，一切外键概念必须在应用层解决

## 十二、工程结构

### 12.1 应用分层
```
├── controller    // 控制层，接收请求
├── service       // 业务逻辑层
│   ├── impl      // 实现类
├── manager       // 通用业务处理层（可选）
├── dao/mapper    // 数据访问层
├── model/entity  // 数据模型
│   ├── dto       // 数据传输对象
│   ├── vo        // 视图对象
│   ├── bo        // 业务对象
```

### 12.2 二方库规约
- 定义 GAV 遵从以下规则：GroupId 格式为 com.{公司/BU}.业务线.子业务线
- 二方库的新增或升级，保持除功能点之外的其它 jar 包仲裁结果不变

---

## 参考资料
- 《阿里巴巴 Java 开发手册》官方 PDF（最新版）
- GitHub 仓库：[阿里巴巴 p3c 仓库](https://github.com/alibaba/p3c)
- IDEA 插件：Alibaba Java Coding Guidelines
