TensorFVM 架构与扩展接口
========================

本文描述当前代码采用的模块边界与扩展约定。此项重构是渐进式第一阶段：
先统一二维求解器后端注册和有限体积网格契约，同时保留现有命令行、
``SolverConfig`` 和 ``SimpleSolver`` 调用方式；配置文件格式、案例生命周期、
输出/后处理、三维求解器及可插拔数值模型还未全部抽离为独立子系统。
因此它是可扩展基础，而不是已经完成的大型 CFD 平台。

运行结构
--------

一次二维求解的关键路径是::

    应用/案例 -> SolverConfig -> backend registry -> solver factory
                                            -> mesh factory -> Mesh2D 验证

* ``solver.py`` 保持已有用户 API，并负责通用设置及 ``SolverConfig`` 校验。
* ``backend_registry.py`` 是后端能力与工厂的单一注册点。后端规则（网格
  限制、瞬态/湍流/入口能力、圆柱约束、参考长度等）随注册项维护，不再
  通过 ``SimpleSolver`` 中的后端名称列表路由。内置实现采用延迟导入。
* ``mesh_api.py`` 定义结构化/非结构化网格均可满足的 ``Mesh2D`` 协议，
  并在网格工厂边界验证几何、拓扑、边界面标记及必需边界组。
* ``body_fitted.py`` 包含当前共用的贴体网格及同位网格 SIMPLE/SA/URANS
  算法；``multi_element.py`` 负责 Gmsh v2 读取及三段翼网格。
* 算例参数、公开基准输入及算例专属 CLI 仍留在既有 benchmark 模块中；
  逐步分离这些职责属于后续工作。

后端能力与兼容性
----------------

``SolverBackend`` 描述每种后端的 ``solver_factory``、``mesh_factory``、
结构性及网格尺寸约束、算法能力、边界 mask 需求和特征长度规则。
已注册内置名称为 ``cartesian``、``body-fitted``、``c-grid``、
``flat-plate``、``three-element``。现有名称、配置字段和调用形式继续有效。
注册项中 ``solver_factory=None`` 专用于由 ``SimpleSolver`` 提供的既有
笛卡尔交错 SIMPLE；新的非笛卡尔后端应明确提供自己的求解器工厂。

其他 Python 包可以在应用启动时注册实现（注册表是进程内的；当前不自动
扫描 Python entry points）::

    from tensorfvm import SolverBackend, register_backend
    from my_cfd_package import MyMesh, MySolver

    register_backend(SolverBackend(
        name="my-unstructured",
        solver_factory=MySolver,
        mesh_factory=MyMesh,
        structured=False,
        min_nx=1,
        min_ny=1,
        supports_transient=True,
        supports_spalart_allmaras=False,
        required_mesh_masks=("inlet", "outlet", "wall", "far-field"),
        reference_length="length",
    ))

之后可用 ``SolverConfig(mesh_type="my-unstructured", ...)`` 构造配置，
并沿用 ``SimpleSolver(config)`` 工厂式兼容入口。重复注册默认报错；测试或
应用显式覆盖已有名称时可传 ``replace=True``。扩展的 mesh factory 应返回
符合下述协议的网格，求解器 factory 接收配置并返回实现所需求解接口的对象。
工厂及算法能力需要由扩展包自行配套；注册只负责路由和基础配置校验。
自定义贴体/非结构求解器可在初始化时调用 ``build_mesh(config)``，统一实例化
其注册网格工厂并执行 ``Mesh2D`` 运行时校验。

Mesh2D 网格契约
---------------

``Mesh2D`` 是 structural protocol，不要求继承 TensorFVM 基类。所有几何
张量使用 PyTorch；Structured 网格可保留二维/三维的场形状，非结构场可
展平或采用 ``(1, ncell)`` 形状。求解器使用以下字段：

* ``vertices``：节点坐标；非结构网格通常为 ``(nvertex, 2)``，结构网格可保留
  ``(ny+1, nx+1, 2)`` 等维度，最后一维必须为 x/y 坐标；
* ``centers``、``volumes``：单元中心和面积，元素数必须一致，面积为正；
* ``field_shape``：二维场结果的形状（结构网格如 ``(ny, nx)``，非结构网格
  如 ``(1, ncell)``），其乘积必须等于单元数；
* ``owner``、``neighbor``：每个面的所有者和邻居单元编号；边界面邻居为 -1；
* ``face_vertices``：每个面的两个端点，形状 ``(nface, 2, 2)``；
* ``face_centers``、``face_area_vectors``、``face_lengths``、``face_normals``：
  面中心、面积向量、面长和单位法向；面积向量必须由 owner 指向外侧；
* ``interior``、``boundary``：与邻接数组一致的面分类；
* ``masks``：由边界名称到一维 bool 面掩码的映射；边界掩码不得选中内部面。

``validate_mesh2d`` 会校验必需字段、形状、索引范围、正体积/正面长、有限
几何数据、法向/面积向量一致性及后端要求的边界组。边界物理条件仍由具体
求解器解释；不应把不同案例的边界策略隐式塞进通用拓扑对象。

约定与下一阶段
--------------

新增后端时应优先通过能力元数据扩展配置校验和工厂注册，而非继续增加
``mesh_type`` 分支；网格生成器负责拓扑与几何，求解器负责离散、边界条件
和收敛策略，算例模块负责参数和基准输入，IO 模块负责稳定的结果 schema。
下一阶段可将配置拆为通用物理参数、离散/算法选项、时间控制和案例参数，
再明确 ``Solver``、``Mesh``、``Field``、``BoundaryCondition``、``ResultWriter``
等长期接口，并添加插件发现、版本化 schema、数据集和求解器集成测试。

30P30N 仍作为现有多单元算例及回归样例；目前编写或运行这套框架不等价于
公开基准数值验证通过。真实 Gmsh 网格生成和完整流场对比需在具备 Gmsh、
PyTorch 及基准网格/工况的环境中单独完成。
