"""Executable check of the key validation of the options readers (chapter 1, _options.qmd).

What is checked: check_keys (src/input/check_keys.cpp at snapy@73c45a7c8c991209924e1137992352c335575062),
ported on Python dicts in place of YAML nodes, with the message format of the TORCH_CHECK. Claims C1-C3.

Run: python3 options_check.py   (numpy only; exits with the number of failed claims)
"""
import sys


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def check_keys(node, path, keys, other="", other_keys=()):
    if not isinstance(node, dict):
        return None                              # a non-map node is not checked
    for key in node:
        if key not in keys and key not in other_keys:
            msg = "unknown key '%s/%s'. Valid keys: %s" % (path, key, ", ".join(keys))
            msg += ("; read by " + other + ": " if other else "") + ", ".join(other_keys) + "."
            raise ValueError(msg)
    return None


INTG = ["implicit-scheme", "implicit-advection-cfl", "shear-cfl", "implicit-precision",
        "implicit-fallback-fp64", "execution", "execution-strict"]  # src/implicit/implicit_hydro.cpp at the pin
PYHARP = ["type", "cfl", "tlim", "nlim", "ncycle_out", "verbose"]


def check_C1_non_map_is_ignored():
    ok = all(check_keys(n, "integration", INTG) is None for n in ([1, 2], "rk3", 3.0, None))
    return report("C1", ok, "a sequence, a scalar or a null node passes unchecked", "4 of 4 passed")


def check_C2_unknown_key_names_the_valid_ones():
    try:
        check_keys({"type": "rk3", "cfll": 0.9}, "integration", INTG, "pyharp", PYHARP)
        msg = None
    except ValueError as e:
        msg = str(e)
    ok = msg is not None and "integration/cfll" in msg and "implicit-scheme" in msg and "read by pyharp" in msg
    return report("C2", ok, "an unknown key is refused with its path, the valid keys and the delegated reader's keys",
                  repr(msg))


def check_C3_delegated_keys_pass():
    try:
        check_keys({"type": "rk3", "cfl": 0.9, "implicit-scheme": 9}, "integration", INTG, "pyharp", PYHARP)
        ok = True
    except ValueError:
        ok = False
    return report("C3", ok, "keys of this reader and of the delegated reader both pass", "accepted: %s" % ok)


def main():
    checks = [check_C1_non_map_is_ignored, check_C2_unknown_key_names_the_valid_ones, check_C3_delegated_keys_pass]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
