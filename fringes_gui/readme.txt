before publishing new version to pypi:
    # pytest
    version
    lock
    black
    push
    build
    publish

when adding new parameter:
    params.py
    setter.py
        set_visibility()
        set_param()
            delattr
        update_parameter_tree
