import unittest

from compiler import *


class SymbolTableTest(unittest.TestCase):
    def setUp(self):
        self.test_symbol_table = SymbolTable()

    def test_define(self):
        table = self.test_symbol_table
        for kind in ["static", "field", "local", "argument"]:
            self.assertEqual(table.var_count(kind), 0)
        table.define("x", INT, "static")
        self._assertOnlyVarInSymbolTable("x", INT, "static")
        table.start_subroutine()
        self.assertEqual(table.var_count("local"), 0)
        self.assertEqual(table.var_count("argument"), 0)
        self._assertOnlyVarInSymbolTable("x", INT, "static")
        table.define("local", CHAR, "local")
        self.assertEqual(table.var_count("local"), 1)
        table.define("m", CHAR, "local")
        self.assertEqual(table.var_count("local"), 2)

    def test_kind_of(self):
        table = self.test_symbol_table

        def kind_of_raises_error():
            return table.kind_of("x")
        self.assertRaises(
            CompileException,
            kind_of_raises_error,
        )
        table.define("x", INT, "static")
        self.assertEqual(table.kind_of("x"), "static")

    def _assertOnlyVarInSymbolTable(self, var, _type, kind):
        table = self.test_symbol_table
        self.assertEqual(table.var_count(kind), 1)
        self.assertIs(table.kind_of(var), kind)
        self.assertIs(table.type_of(var), _type)
        self.assertEqual(table.index_of(var), 0)


class WriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.writer = Writer(open('..\\bar.baz.vm', 'w'))

    def test_base_name(self):
        self.assertEqual(self.writer.base_name, 'bar.baz')


if __name__ == '__main__':
    unittest.main()
