import os
import re
import sys
from typing import NamedTuple
import logging
import enum

IN_FILE_EXT = '.jack'
OUT_FILE_EXT = '_test.xml'
NEWLINE = '\n'
INDENT_NUM_SPACES = 2
# Jack Lexical elements
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

UNARY_OP = {MINUS, TILDE}   # faster for in operator
OP = {PLUS, MINUS, ASTERISK, FORWARD_SLASH, AMPERSAND, PIPE, LESS_THAN, GREATER_THAN, EQUAL_SIGN}
KEYWORD_CONSTANT = {TRUE, FALSE, NULL, THIS}


class ParseException(Exception):
    pass


class Token(NamedTuple):
    type: str
    value: str
    line_number: int


# Module 2: JackTokenizer
class JackTokenizer:
    # Note, order of these specifications matter
    tokens_specifications = {
        'comment': r'//.*|/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/',
        'space': r'[ \t]+',
        'newline': r'\n',
        KEYWORD: '|'.join([
            'class', 'constructor', 'function',
            'method', 'field', 'static', 'var', 'int',
            'char', 'boolean', 'void', 'true', 'false',
            'null', 'this', 'let', 'do', 'if', 'else',
            'while', 'return'
        ]),
        SYMBOL: '|'.join([
            r'\{', r'\}', r'\(', r'\)', r'\[', r'\]', r'\.',
            r'\,', r'\;', r'\+', r'\-', r'\*', r'\/', r'\&',
            r'\|', r'\<', r'\>', r'\=', r'\~',
        ]),
        INT_CONSTANT: r'\d+',
        STR_CONSTANT: r'"[^"\n]+"',  # unicode characters
        IDENTIFIER: r'[a-zA-Z_]{1}[a-zA-Z0-9_]*',  # must be after keywords, in python re site, considered keyword as
        # part of ID pattern newline=r'\n',
        'mismatch': r'.',  # any other character
    }
    jack_token = re.compile(
        '|'.join(
            [r'(?P<{}>{})'.format(token, specification)
             for token, specification in tokens_specifications.items()]))

    def __init__(self, in_stream):
        self.in_stream = in_stream

    def start_tokenizer(self):
        line_number = 0
        for m in self.jack_token.finditer(self.in_stream):
            token_type = m.lastgroup
            token_value = m.group(token_type)
            if token_type == 'integerConstant':
                token_value = int(token_value)
            elif token_type == 'newline':
                line_number += 1
                continue
            elif token_type in ('space', 'comment'):
                continue
            elif token_type == 'mismatch':
                raise ParseException(
                    f'got wrong jack token: {token_value} in line {line_number}')
            yield Token(token_type, token_value, line_number)


# TODO: change to output the vm code (no more xml)
# Module 5: CompilationEngine
class CompilationEngine:
    special_xml = {
        LESS_THAN: '&lt;',
        GREATER_THAN: '&gt;',
        AMPERSAND: '&amp;',
        DOUBLE_QUOTES: '&quot;'
    }

    def __init__(self, tokens_stream, out_stream):
        """ initialize the compilation engine which parses tokens from tokensStream and write in outFileStream
        INVARIANT: current_token is the token we are handling now given _eat() is last to run in handling it
        Args:
            tokens_stream (Generator): Generator of jack tokens
            out_stream (stream): file to write parsed jack code into
        """
        self.tokens_stream = tokens_stream
        self.out_stream = out_stream
        self.current_token = None
        self.indent_level = 0

    @property
    def current_token_value(self):
        return self.current_token.value

    @property
    def current_token_type(self):
        return self.current_token.type

    def reindent(self, indent=1):
        self.indent_level += indent

    def deindent(self, indent=1):
        self.indent_level -= indent

    def _write_tag_value(self, tag, value):
        """writes xml tagged jack token to outFileStream
        Args:
            tag (str): type of token
            value (str | integer): value of token
        """
        value = self.special_xml.get(value, value)
        indent = ' ' * INDENT_NUM_SPACES * self.indent_level
        self.out_stream.write(f'{indent}<{tag}> {value} </{tag}>{NEWLINE}')

    def _write_open_tag(self, tag):
        """writes xml open tag with given tag
        Args:
            tag (str): xml tag
        """
        indent = ' ' * INDENT_NUM_SPACES * self.indent_level
        self.out_stream.write(f'{indent}<{tag}>{NEWLINE}')

    def _write_close_tag(self, tag):
        """writes xml close tag with given tag
        Args:
            tag (str): xml tag
        """
        indent = ' ' * INDENT_NUM_SPACES * self.indent_level
        self.out_stream.write(f'{indent}</{tag}>{NEWLINE}')

    def _eat(self, s):
        """advance to next token if given string is same as the current token, otherwise raise error
        Args:
            s (str): string to match current token against
        Raises:
            ParseException: in case no match
        """
        if s == self.current_token_value or \
                (s == self.current_token_type and s in {INT_CONSTANT, STR_CONSTANT, IDENTIFIER}):
            try:
                self.current_token = next(self.tokens_stream)
            except StopIteration:
                if s != RIGHT_BRACE:  # last token
                    raise ParseException(f'Error, reached end of file\n{str(self.current_token)}')
        else:
            raise ParseException(
                f'Got wrong token in line {self.current_token.line_number}: '
                f'{self.current_token_value}, expected: {s!r}\n{str(self.current_token)}')

    def compile_class(self):
        """Starting point in compiling a jack source file
        """
        # first token
        try:
            self.current_token = self.current_token or next(self.tokens_stream)
        except StopIteration:  # jack source file is empty
            return

        # <class>
        self._write_open_tag(CLASS)
        self.reindent()

        # class
        self._write_tag_value(KEYWORD, self.current_token_value)
        self._eat(CLASS)

        # className
        self._write_tag_value(IDENTIFIER, self.current_token_value)
        self._eat(IDENTIFIER)

        # {
        self._write_tag_value(SYMBOL, self.current_token_value)
        self._eat(LEFT_BRACE)

        # classVarDec*
        while self.current_token_value in {STATIC, FIELD}:
            self.compile_class_var_dec()

        # subroutineDec*
        while self.current_token_value in {CONSTRUCTOR, FUNCTION, METHOD}:
            self.compile_subroutine_dec()

        # }
        self._write_tag_value(SYMBOL, self.current_token_value)
        self._eat(RIGHT_BRACE)

        # </class>
        self.deindent()
        self._write_close_tag(CLASS)

    def compile_class_var_dec(self):
        """compile a jack class variable declarations
        ASSUME: only called if current token's value is static or field
        """
        # <classVarDec>
        self._write_open_tag('classVarDec')
        self.reindent()

        # field | static
        self._write_tag_value(KEYWORD, self.current_token_value)
        if self.current_token_value in (STATIC, FIELD):
            self._eat(self.current_token_value)

        # varName
        self._handle_type_var_name()

        # </classVarDec>
        self.deindent()
        self._write_close_tag('classVarDec')

    def _handle_type_var_name(self):
        # type
        if self.current_token_value in {INT, CHAR, BOOLEAN}:
            self._write_tag_value(KEYWORD, self.current_token_value)
            self._eat(self.current_token_value)
        elif self.current_token_type == IDENTIFIER:
            self._write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)

        # varName (, varName)*;
        while True:
            self._write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)
            if self.current_token_value == SEMI_COLON:
                break
            self._write_tag_value(SYMBOL, COMMA)
            self._eat(COMMA)
        self._write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

    def compile_subroutine_dec(self):
        """compile a jack class subroutine declarations
        ASSUME: only called if current token's value is constructor, function or method
        """
        # <subroutineDec>
        self._write_open_tag('subroutineDec')
        self.reindent()

        # constructor | function | method
        self._write_tag_value(KEYWORD, self.current_token_value)
        if self.current_token_value in (CONSTRUCTOR, FUNCTION, METHOD):
            self._eat(self.current_token_value)

        # void | type
        if self.current_token_value in (VOID, INT, CHAR, BOOLEAN):
            self._write_tag_value(KEYWORD, self.current_token_value)
            self._eat(self.current_token_value)
        elif self.current_token_type == IDENTIFIER:
            self._write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)

        # subroutineName
        self._write_tag_value(IDENTIFIER, self.current_token_value)
        self._eat(IDENTIFIER)

        # (
        self._write_tag_value(SYMBOL, self.current_token_value)
        self._eat(LEFT_PAREN)

        # parameterList
        self.compile_parameter_list()

        # )
        self._write_tag_value(SYMBOL, self.current_token_value)
        self._eat(RIGHT_PAREN)

        # subroutineBody
        self.compile_subroutine_body()

        # </subroutineDec>
        self.deindent()
        self._write_close_tag('subroutineDec')

    def compile_parameter_list(self):
        """compile a jack parameter list which is 0 or more list
        """
        # <parameterList>
        self._write_open_tag('parameterList')
        self.reindent()

        # ((type varName) (, type varName)*)?
        while True:
            if self.current_token_value in {INT, CHAR, BOOLEAN}:
                self._write_tag_value(KEYWORD, self.current_token_value)
                self._eat(self.current_token_value)
            elif self.current_token_type == IDENTIFIER:
                self._write_tag_value(IDENTIFIER, self.current_token_value)
                self._eat(IDENTIFIER)
            else:
                break

            self._write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)
            if not self.current_token_value == COMMA:
                break
            self._write_tag_value(SYMBOL, COMMA)
            self._eat(COMMA)

        self.deindent()
        self._write_close_tag('parameterList')

    def compile_subroutine_body(self):
        """compile a jack subroutine body which is varDec* statements
        """
        # <subroutineBody>
        self._write_open_tag('subroutineBody')
        self.reindent()

        # {
        self._write_tag_value(SYMBOL, LEFT_BRACE)
        self._eat(LEFT_BRACE)

        while self.current_token_value == VAR:  # order matters, simplify
            self.compile_var_dec()

        self.compile_statements()

        # }
        self._write_tag_value(SYMBOL, RIGHT_BRACE)
        self._eat(RIGHT_BRACE)

        # </subroutineBody>
        self.deindent()
        self._write_close_tag('subroutineBody')

    def compile_var_dec(self):
        """compile jack variable declaration, varDec*, only called if current token is var
        add the variable to symbol table
        """
        # <varDec>
        self._write_open_tag('varDec')
        self.reindent()

        # VAR
        self._write_tag_value(KEYWORD, self.current_token_value)
        self._eat(VAR)

        # type varName (',' varName)*;
        self._handle_type_var_name()

        # </varDec>
        self.deindent()
        self._write_close_tag('varDec')

    def compile_statements(self):
        """
        match the current token value to the matching jack statement
        """
        # <statements>
        self._write_open_tag('statements')
        self.reindent()

        while self.current_token_value in {LET, IF, WHILE, DO, RETURN}:
            {
                LET: self.compile_let_statement,
                IF: self.compile_if_statement,
                WHILE: self.compile_while_statement,
                DO: self.compile_do_statement,
                RETURN: self.compile_return_statement,
            }[self.current_token_value]()

        # </statements>
        self.deindent()
        self._write_close_tag('statements')

    def compile_let_statement(self):
        """
        compile jack let statement
        """
        # <letStatement>
        self._write_open_tag('letStatement')
        self.reindent()

        # let
        self._write_tag_value(KEYWORD, LET)
        self._eat(LET)

        # varName
        self._write_tag_value(IDENTIFIER, self.current_token_value)
        self._eat(IDENTIFIER)

        # ( '[' expression ']')?
        if self.current_token_value == LEFT_BRACKET:
            self._write_tag_value(SYMBOL, LEFT_BRACKET)
            self._eat(LEFT_BRACKET)
            self.compile_expression()
            self._write_tag_value(SYMBOL, RIGHT_BRACKET)
            self._eat(RIGHT_BRACKET)

        # =
        self._write_tag_value(SYMBOL, EQUAL_SIGN)
        self._eat(EQUAL_SIGN)

        # expression
        self.compile_expression()

        # ;
        self._write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

        # <letStatement>
        self.deindent()
        self._write_close_tag('letStatement')

    def compile_if_statement(self):
        """
        compile jack if statement
        """

        # <ifStatement>
        self._write_open_tag('ifStatement')
        self.reindent()

        # if
        self._write_tag_value(KEYWORD, IF)
        self._eat(IF)

        # (expression)
        self._handle_expr_or_expr_list_within_paren(self.compile_expression)

        # {statements}
        self._handle_statements_within_braces()

        if self.current_token_value == ELSE:
            self._write_tag_value(KEYWORD, ELSE)
            self._eat(ELSE)
            self._handle_statements_within_braces()

        # <ifStatement>
        self.deindent()
        self._write_close_tag('ifStatement')

    def _handle_expr_or_expr_list_within_paren(self, compile_function):
        # (
        self._write_tag_value(SYMBOL, LEFT_PAREN)
        self._eat(LEFT_PAREN)
        # compile_expression or compile_expression_list
        compile_function()
        # )
        self._write_tag_value(SYMBOL, RIGHT_PAREN)
        self._eat(RIGHT_PAREN)

    def _handle_statements_within_braces(self):
        # {
        self._write_tag_value(SYMBOL, LEFT_BRACE)
        self._eat(LEFT_BRACE)
        # statements
        while self.current_token_value in {LET, IF, WHILE, DO, RETURN}:
            self.compile_statements()
        # }
        self._write_tag_value(SYMBOL, RIGHT_BRACE)
        self._eat(RIGHT_BRACE)

    def compile_while_statement(self):
        """
        compile jack while statement
        """

        # <whileStatement>
        self._write_open_tag('whileStatement')
        self.reindent()

        # while
        self._write_tag_value(KEYWORD, WHILE)
        self._eat(WHILE)

        # (expression)
        self._handle_expr_or_expr_list_within_paren(self.compile_expression)

        # {statements}
        self._handle_statements_within_braces()

        # <whileStatement>
        self.deindent()
        self._write_close_tag('whileStatement')

    def compile_do_statement(self):
        """
        compile jack do statement
        """

        # <doStatement>
        self._write_open_tag('doStatement')
        self.reindent()

        # do
        self._write_tag_value(KEYWORD, DO)
        self._eat(DO)

        #  subroutineName | (className | varName)'.'subroutineName
        self._write_tag_value(IDENTIFIER, self.current_token_value)
        self._eat(IDENTIFIER)
        # check if '.'
        if self.current_token_value == DOT:
            self._write_tag_value(SYMBOL, DOT)
            self._eat(DOT)
            self._write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)

        # (expressionList)
        self._handle_expr_or_expr_list_within_paren(self.compile_expression_list)

        # ;
        self._write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

        # </doStatement>
        self.deindent()
        self._write_close_tag('doStatement')

    def compile_return_statement(self):
        """
        compile jack return statement
        """
        # <returnStatement>
        self._write_open_tag('returnStatement')
        self.reindent()

        # return
        self._write_tag_value(KEYWORD, RETURN)
        self._eat(RETURN)

        # expression?
        if self.current_token_value != SEMI_COLON:
            self.compile_expression()

        # ;
        self._write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

        # </returnStatement>
        self.deindent()
        self._write_close_tag('returnStatement')

    def compile_expression(self):
        """
        compile jack expression
        """
        """
        if exp is a number n:
            output "push n"
        if exp is a variable var:
            output "push var"
        if exp is "exp1 op exp2":
            codeWrite(exp1)
            codeWrite(exp2)
            output "op"
        if exp is "f(exp1, exp2, ...)":
            codeWrite(exp1),
            codeWrite(exp2),...,
            output "call f"
        """
        # <expression>
        self._write_open_tag('expression')
        self.reindent()

        # term
        self.compile_term()

        # (op term)*
        while self.current_token_value in OP:
            self._write_tag_value(SYMBOL, self.current_token_value)
            self._eat(self.current_token_value)
            self.compile_term()

        # </expression>
        self.deindent()
        self._write_close_tag('expression')

    def compile_term(self):
        """
        compile jack term
        """

        # <term>
        self._write_open_tag('term')
        self.reindent()

        if self.current_token_type == INT_CONSTANT:
            self._write_tag_value('integerConstant', self.current_token_value)
            self._eat(INT_CONSTANT)
        elif self.current_token_type == STR_CONSTANT:
            self._write_tag_value('stringConstant', self.current_token_value.strip(DOUBLE_QUOTES))
            self._eat(STR_CONSTANT)
        elif self.current_token_value in KEYWORD_CONSTANT:
            self._write_tag_value(KEYWORD, self.current_token_value)
            self._eat(self.current_token_value)
        elif self.current_token_value in UNARY_OP:
            self._write_tag_value(SYMBOL, self.current_token_value)
            self._eat(self.current_token_value)
            self.compile_term()
        elif self.current_token_value == LEFT_PAREN:  # '(' expression ')'
            self._handle_expr_or_expr_list_within_paren(self.compile_expression)
        else:  # identifier
            current_token_value = self.current_token_value
            self._eat(IDENTIFIER)
            next_token_value = self.current_token_value

            # varName'[' expression ']'
            if next_token_value == LEFT_BRACKET:
                self._write_tag_value(IDENTIFIER, current_token_value)
                self._write_tag_value(SYMBOL, LEFT_BRACKET)
                self._eat(LEFT_BRACKET)
                self.compile_expression()
                self._write_tag_value(SYMBOL, RIGHT_BRACKET)
                self._eat(RIGHT_BRACKET)
            # subroutineCall: foo.bar(expressionList) | Foo.bar(expressionList)
            elif next_token_value == DOT:
                self._write_tag_value(IDENTIFIER, current_token_value)
                self._write_tag_value(SYMBOL, DOT)
                self._eat(DOT)
                self._write_tag_value(IDENTIFIER, self.current_token_value)
                self._eat(IDENTIFIER)
                self._handle_expr_or_expr_list_within_paren(self.compile_expression_list)
            # subroutineCall: bar(expressionList)
            elif next_token_value == LEFT_PAREN:
                self._write_tag_value(IDENTIFIER, current_token_value)
                self._handle_expr_or_expr_list_within_paren(self.compile_expression_list)
            # foo
            else:
                self._write_tag_value(IDENTIFIER, current_token_value)

        # </term>
        self.deindent()
        self._write_close_tag('term')

    def compile_expression_list(self):
        """
        compile jack expression list
        """
        # <expressionList>
        self._write_open_tag('expressionList')
        self.reindent()

        # (expression (',' expression)*)?
        if not self.current_token_value == RIGHT_PAREN:
            while True:
                self.compile_expression()
                if not self.current_token_value == COMMA:
                    break
                self._write_tag_value(SYMBOL, COMMA)
                self._eat(COMMA)

        # </expressionList>
        self.deindent()
        self._write_close_tag('expressionList')


# TODO: implement
# Module 4: VMWriter, generates VM code
class VMWriter:

    # in java sense, should initialize the .vm file object
    def __init__(self, out_stream):
        self.out_stream = out_stream

    # write a vm push command
    def write_push(self, segment, index):
        pass

    # write a vm pop command
    def write_pop(self, segment, index):
        pass

    # write a vm arithmetic-logical command
    def write_arithmetic(self, vm_command):
        pass

    # writes a vm label command
    def write_label(self, string):
        pass

    # writes a vm goto command
    def write_goto(self, string):
        pass

    # writes a vm if-goto command
    def write_if_goto(self, string):
        pass

    # writes a vm call command
    def write_call(self, name: str, n_args: int):
        pass

    # write a vm function
    def write_function(self, name: str, n_locals: int):
        pass

    # writes a vm return command
    def write_return(self):
        pass

    # closes the output file
    def close(self):
        pass


# TODO: implement
# Module 3: Symbol Table, we need 2: class-level and subroutine-level
""" Symbol tables
name    type    kind        #
x       int     field       0
y       int     field       1
"""


# helper enums
class JType(enum.Enum):
    JInt = 1
    JChar = 2
    JBoolean = 3
    JCustom = 4  # class name


class JVarKind(enum.Enum):
    field = 1
    static = 2
    local = 3
    argument = 4


# class JVarScope(enum.Enum):
#     class_level = 1
#     subroutine_level = 2


class JVariable(NamedTuple):
    name: str
    type: JType
    kind: JVarKind
    index: int

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, JVariable):
            return False
        return self.name == o.name and self.type == o.type and self.kind == o.kind


class SymbolTable:
    def __init__(self):
        _class_level_data = []
        _subroutine_data = []

    # starts a new subroutine scope
    def start_subroutine(self):
        pass

    # defines a new identifier of given parameters and assign it a running index
    def define(self, name: str, _type: JType, kind: JVarKind):
        pass

    # return the number of variables of the given kind
    def var_count(self, kind: JVarKind):
        pass

    # return the kind of the identifier var
    def kind_of(self, var: str):
        pass

    # return the type of the identifier var
    def type_of(self, var: str):
        pass

    # return the index of the identifier var
    def index_of(self, var: str):
        pass


# TODO: adjust to output vm files (maybe keep xml too)
# Module 1: Jack compiler (ui)
def handle_file(path):
    logging.info(f'Parsing {path}')
    out_file_path = path.replace(IN_FILE_EXT, OUT_FILE_EXT)
    with open(path) as inFileStream:
        with open(out_file_path, 'w') as outFileStream:
            tokens_stream = JackTokenizer(inFileStream.read()).start_tokenizer()
            compilation_engine = CompilationEngine(tokens_stream, outFileStream)
            compilation_engine.compile_class()
    logging.info(f'generated {out_file_path}')


def handle_dir(path):
    for f in os.listdir(path):
        if f.endswith(IN_FILE_EXT):
            handle_file(os.path.join(path, f))


def main(args=None):
    if args is None:
        args = sys.argv[1:]
        if not args:
            logging.error('Usage: SyntaxAnalyzer.py <path-to-jack-file-or-directory-of-source-code>')

    for file in args:
        if os.path.isfile(file):
            handle_file(file)
        elif os.path.isdir(file):
            handle_dir(file)
        else:
            logging.error(f'{", ".join(args)} are not jack source files')
            return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(message)s',
    )
    sys.exit(main())
