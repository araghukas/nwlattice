def get_info(type_name: str = None) -> None:
    """
    Print information about nanowire classes in `nwlattice/nw.py`.

    If `type_name` is specified then only information about that class is
    printed.
    """
    from inspect import getmembers
    from difflib import get_close_matches
    import nwlattice.base as base
    import nwlattice.nw as nw
    obj_list = getmembers(nw)

    nw_class_list = []
    for o in obj_list:
        if o[1] is base.ANanowireLattice:
            continue
        try:
            if issubclass(o[1], base.ANanowireLattice):
                nw_class_list.append(o[1])
        except TypeError:
            pass

    type_names = []
    for nw_class in nw_class_list:
        name = str(nw_class).lstrip("<class '").rstrip("'>")
        name = name.split('.')[-1]
        type_names.append(name)

    if type_name:
        type_name_index = None
        exists = False
        for i, tn in enumerate(type_names):
            if type_name == tn:
                exists = True
                type_name_index = i
                break

        if not exists:
            print("nanowire type `%s` not found in nwlattice.nw" % type_name)
            matches = get_close_matches(type_name, type_names)
            if matches:
                print("did you mean: `%s`?" % matches[0])
            return

        print(type_name)
        print(nw_class_list[type_name_index].__doc__[1:])
        print(nw_class_list[type_name_index].__init__.__doc__[1:])

    else:
        for type_name, nw_class in zip(type_names, nw_class_list):
            print(type_name)
            print(nw_class.__doc__[1:])
            print(nw_class.__init__.__doc__[1:])
        return


__version__ = "31Jan2021"
