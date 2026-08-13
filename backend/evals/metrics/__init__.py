"""Pure, unit-testable scoring functions.

Every function here takes plain data — event lists, strings, floats — and
returns plain data. None of them call a model, a store, or the network, which
is what makes them checkable without a paid run: build the event list by hand,
call the function, assert on the result.
"""
