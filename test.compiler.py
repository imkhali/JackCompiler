import unittest
from compiler import *


class SymbolTableTest(unittest.TestCase):
    def setUp(self):
        self.test_symbol_table = SymbolTable()

    def test_define(self):
        table = self.test_symbol_table
        for kind in JVarKind:
            self.assertEqual(table.var_count(kind), 0)
        table.define("x", JType.JInt, JVarKind.static)
        self._assertOnlyVarInSymbolTable("x", JType.JInt, JVarKind.static)
        table.start_subroutine()
        self.assertEqual(table.var_count(JVarKind.local), 0)
        self.assertEqual(table.var_count(JVarKind.argument), 0)
        self._assertOnlyVarInSymbolTable("x", JType.JInt, JVarKind.static)
        table.define("local", JType.JChar, JVarKind.local)
        self.assertEqual(table.var_count(JVarKind.local), 1)

    def _assertOnlyVarInSymbolTable(self, var, _type, kind):
        table = self.test_symbol_table
        self.assertEqual(table.var_count(kind), 1)
        self.assertIs(table.kind_of(var), kind)
        self.assertIs(table.type_of(var), _type)
        self.assertEqual(table.index_of(var), 0)


if __name__ == '__main__':
    unittest.main()
