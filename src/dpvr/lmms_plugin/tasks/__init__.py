"""空的 tasks 子包，只为满足 lmms-eval 0.5.0 的插件约定。

``lmms_eval/__main__.py:435`` 对 ``LMMS_EVAL_PLUGINS`` 里的每个名字无条件执行

    importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]

而 ``scripts/revision/eval_lmms.py`` 会无条件追加 ``dpvr.lmms_plugin``。本包只注册
模型（``models/``），原本没有 ``tasks/``，于是 ``find_spec`` 返回 ``None``，
取 ``[0]`` 抛 ``AttributeError``，所有评测在启动阶段就死。

加上本文件后名字可以解析。目录保持为空：那段代码在 ``TaskManager`` 建好之后才把
路径追加进 ``args.include_path``，之后无人再读，是死代码，所以这里放什么都不会
被扫描。真要加自定义任务时再在这里建模块即可。
"""
