window.FORVIA_CONSOLE_MODULES = [
  {
    id: "database",
    icon: "🗄",
    title: "数据库",
    desc: "新建/更新数据库、配置标签规则、查看数据标签分布。",
    url: "http://127.0.0.1:8001/#database:operation",
    primary: true,
  },
  {
    id: "label",
    icon: "🎧",
    title: "数据标注",
    desc: "任务、列表总览、逐条标注、高级分析、Prototype 和导出。",
    url: "http://127.0.0.1:8012/",
  },
  {
    id: "train",
    icon: "🧠",
    title: "模型项目",
    desc: "项目列表、配置数据集、配置模型、提取特征和训练结果。",
    url: "http://127.0.0.1:8001/",
  },
  {
    id: "predict",
    icon: "📤",
    title: "模型推理",
    desc: "选择已训练项目，使用最新模型对单个 TDMS 或文件夹执行推理。",
    url: "http://127.0.0.1:8001/#predict",
  },
];
