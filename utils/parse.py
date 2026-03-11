import importlib


CONTROLLER_REGISTRY = {
    "GtController": ("controller.gtcontroller", "GtController"),
    "ModelController": ("controller.modelcontroller", "ModelController"),
}


MANIPULATION_REGISTRY = {
    "OpenBottleManipulation": ("manipulation.open_bottle", "OpenBottleManipulation"),
    "OpenMicroWaveManipulation": ("manipulation.open_microwave", "OpenMicroWaveManipulation"),
    "OpenPenManipulation": ("manipulation.open_pen", "OpenPenManipulation"),
    "OpenDoorManipulation": ("manipulation.open_door", "OpenDoorManipulation"),
    "OpenWindowManipulation": ("manipulation.open_window", "OpenWindowManipulation"),
    "OpenPressureCookerManipulation": ("manipulation.open_pc", "OpenPressureCookerManipulation"),
    "OpenCoffeeMachineManipulation": ("manipulation.open_cm", "OpenCoffeeMachineManipulation"),
    "OpenLampManipulation": ("manipulation.open_lamp", "OpenLampManipulation"),
    "OpenSafeManipulation": ("manipulation.open_safe", "OpenSafeManipulation"),
}


def _load_symbol(registry, name):
    module_name, class_name = registry[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def parse_controller(args, env, manipulation, cfg, logger):
    print(args.controller)
    controller_class = _load_symbol(CONTROLLER_REGISTRY, args.controller)
    return controller_class(env, manipulation, cfg, logger)

def parse_manipulation(args, env, cfg, logger):
    manipulation_class = _load_symbol(MANIPULATION_REGISTRY, args.manipulation)
    return manipulation_class(env, cfg, logger)