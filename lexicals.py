# keywords
CLASS = 'class'
CONSTRUCTOR = 'constructor'
FUNCTION = 'function'
METHOD = 'method'
FIELD = 'field'
STATIC = 'static'
VAR = 'var'
INT = 'int'
CHAR = 'char'
BOOLEAN = 'boolean'
VOID = 'void'
TRUE = 'true'
FALSE = 'false'
NULL = 'null'
THIS = 'this'
THAT = 'that'
LET = 'let'
DO = 'do'
IF = 'if'
ELSE = 'else'
WHILE = 'while'
RETURN = 'return'
# Symbols
LEFT_BRACE = '{'
RIGHT_BRACE = '}'
LEFT_PAREN = '('
RIGHT_PAREN = ')'
LEFT_BRACKET = '['
RIGHT_BRACKET = ']'
DOT = '.'
COMMA = ','
SEMI_COLON = ';'
PLUS = '+'
MINUS = '-'
ASTERISK = '*'
FORWARD_SLASH = '/'
AMPERSAND = '&'
PIPE = '|'
LESS_THAN = '<'
GREATER_THAN = '>'
EQUAL_SIGN = '='
TILDE = '~'
# other constants
DOUBLE_QUOTES = '"'
INT_CONSTANT = 'integerConstant'
STR_CONSTANT = 'stringConstant'
IDENTIFIER = 'identifier'
KEYWORD = 'keyword'
SYMBOL = 'symbol'
UNARY_OP = {MINUS, TILDE}  # faster for in operator
OP = {PLUS, MINUS, ASTERISK, FORWARD_SLASH, AMPERSAND, PIPE, LESS_THAN, GREATER_THAN, EQUAL_SIGN}
KEYWORD_CONSTANT = {TRUE, FALSE, NULL, THIS}
SUBROUTINE = 'subroutine'
LOCAL = 'local'
ARGUMENT = 'argument'

VM_OPERATIONS = {
    PLUS: "add",
    MINUS: "sub",
    ASTERISK: "call Math.multiply 2",
    FORWARD_SLASH: "call Math.divide 2",
    EQUAL_SIGN: "eq",
    LESS_THAN: "lt",
    GREATER_THAN: "gt",
    AMPERSAND: "and",
    PIPE: "or",
    TILDE: "not"
}
