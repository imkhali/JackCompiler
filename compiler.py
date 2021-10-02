import logging
import os
import re
import sys
from operator import attrgetter
from typing import NamedTuple, TextIO, Optional

from lexicals import *

# informative labels
IF_FALSE = "IF_FALSE"
IF_TRUE = "IF_TRUE"
WHILE_END = "WHILE_END"
WHILE_EXP = 'WHILE_EXP'

SRC_FILE_EXT = '.jack'
VM_FILE_EXT = '.vm'
NEWLINE = '\n'
INDENT_NUM_SPACES = 2


class ParseException(Exception):
    pass


class CompileException(Exception):
    pass


class Token(NamedTuple):
    type: str
    value: str
    line_number: int


# Module 2: JackTokenizer
class JackTokenizer:
    # Note, order of these specifications matter
    KEYWORDS = {
        'class', 'constructor', 'function',
        'method', 'field', 'static', 'var', 'int',
        'char', 'boolean', 'void', 'true', 'false',
        'null', 'this', 'let', 'do', 'if', 'else',
        'while', 'return'
    }
    tokens_specifications = {
        'comment': r'//.*|/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/',
        'space': r'[ \t]+',
        'newline': r'\n',
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

    @property
    def src_base_name(self):
        return os.path.split(self.in_stream.name)[-1].rpartition('.')[0]

    def start_tokenizer(self):
        line_number = 1
        for m in self.jack_token.finditer(self.in_stream.read()):
            token_type = m.lastgroup
            token_value = m.group(token_type)
            if token_type == 'integerConstant':
                token_value = int(token_value)
            elif token_type == IDENTIFIER and token_value in self.KEYWORDS:
                token_type = KEYWORD
            elif token_type == 'newline':
                line_number += 1
                continue
            elif token_type in ('space', 'comment'):
                continue
            elif token_type == 'mismatch':
                raise ParseException(
                    f'got wrong jack token: {token_value} in line {line_number}')
            yield Token(token_type, token_value, line_number)


# Module 4: VMWriter, generates VM code
class Writer:
    def __init__(self, out_stream: TextIO):
        self.out_stream = out_stream

    @property
    def base_name(self):
        return os.path.split(self.out_stream.name)[-1].rpartition('.')[0]


class VMWriter(Writer):
    def write_push(self, segment, index):
        self.out_stream.write(f"push {segment} {index}{NEWLINE}")

    def write_pop(self, segment, index):
        self.out_stream.write(f"pop {segment} {index}{NEWLINE}")

    def write_arithmetic(self, vm_command):
        self.out_stream.write(f"{vm_command}{NEWLINE}")

    # writes a vm label command
    def write_label(self, label):
        self.out_stream.write(f"label {label}{NEWLINE}")

    # writes a vm goto command
    def write_goto(self, label):
        self.out_stream.write(f"goto {label}{NEWLINE}")

    # writes a vm if-goto command
    def write_if_goto(self, label):
        self.out_stream.write(f"if-goto {label}{NEWLINE}")

    # writes a vm call command
    def write_call(self, name: str, n_args: int):
        self.out_stream.write(f"call {name} {n_args}{NEWLINE}")

    # write a vm function
    def write_function(self, name: str, n_locals: int):
        self.out_stream.write(f"function {name} {n_locals}{NEWLINE}")

    # writes a vm return command
    def write_return(self):
        self.out_stream.write(f"return{NEWLINE}")

    # closes the output file, I guess not needed here, it is java-style
    def close(self):
        pass


# Module 5: CompilationEngine
class CompilationEngine:
    def __init__(self, jack_tokenizer: JackTokenizer, vm_stream: VMWriter):
        """ initialize the compilation engine which parses tokens from tokensStream and write in vm_stream
        INVARIANT: current_token is the token we are handling now given _eat() is last to run in handling it
        Args:
            jack_tokenizer (JackTokenizer): the jack source code tokenizer
            vm_stream (VMWriter): writer of compiled vm code
        """
        self.jack_tokenizer = jack_tokenizer
        self.tokens_stream = jack_tokenizer.start_tokenizer()
        self.vm_stream = vm_stream
        self.symbol_table = SymbolTable()
        self.current_token = None

        # labels indices
        self.labels_indices = {}

    @property
    def current_token_value(self):
        return self.current_token.value

    @property
    def current_token_type(self):
        return self.current_token.type

    @property
    def current_class_name(self):
        return self.jack_tokenizer.src_base_name

    def get_next_label(self, label_prefix):
        index = self.labels_indices.setdefault(label_prefix, -1) + 1
        self.labels_indices[label_prefix] = index
        return f'{label_prefix}{index}'

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
        # class
        self._eat(CLASS)

        # className
        try:
            assert self.current_token_value == self.current_class_name
        except AssertionError:
            raise CompileException(
                f"class {self.current_token_value} does not match filename {self.current_class_name}")
        self._eat(IDENTIFIER)

        # {
        self._eat(LEFT_BRACE)

        # classVarDec*
        while self.current_token_value in {STATIC, FIELD}:
            self.compile_class_var_dec()

        # subroutineDec*
        while self.current_token_value in {CONSTRUCTOR, FUNCTION, METHOD}:
            self.symbol_table.start_subroutine()
            self.compile_subroutine_dec()

        # }
        self._eat(RIGHT_BRACE)

        # </class>

    def compile_class_var_dec(self):
        """compile a jack class variable declarations
        ASSUME: only called if current token's value is static or field
        """
        # <classVarDec>
        # field | static
        kind = None
        if self.current_token_value in (STATIC, FIELD):
            kind = self.current_token_value
            self._eat(self.current_token_value)

        for _type, name in self._handle_type_var_name():
            self.symbol_table.define(name=name, _type=_type, kind=kind)

        # varName
        # </classVarDec>

    def _handle_type_var_name(self):
        """
        :return: generator of jack variable (type, name)
        """
        # type
        _type = None
        if self.current_token_value in {INT, CHAR, BOOLEAN}:
            _type = self.current_token_value
            self._eat(self.current_token_value)
        elif self.current_token_type == IDENTIFIER:
            _type = self.current_token_value
            self._eat(IDENTIFIER)

        # varName (, varName)*;
        while True:
            name = self.current_token_value
            self._eat(IDENTIFIER)
            yield _type, name
            if self.current_token_value == SEMI_COLON:
                break
            self._eat(COMMA)
        self._eat(SEMI_COLON)

    def compile_subroutine_dec(self):
        """compile a jack class subroutine declarations
        ASSUME: only called if current token's value is constructor, function or method
        """
        # <subroutineDec>
        # constructor | function | method
        subroutine_type = self.current_token_value
        if subroutine_type in (CONSTRUCTOR, FUNCTION, METHOD):
            self._eat(subroutine_type)

        # builtin type | className
        if self.current_token_value in (VOID, INT, CHAR, BOOLEAN):
            self._eat(self.current_token_value)
        elif self.current_token_type == IDENTIFIER:
            self._eat(IDENTIFIER)

        # subroutineName
        subroutine_name = self.current_token_value
        self._eat(IDENTIFIER)

        # (
        self._eat(LEFT_PAREN)

        # add this as argument in case the subroutine_type is a method
        if subroutine_type == METHOD:
            self.symbol_table.define(name=THIS, _type=self.current_class_name, kind=ARGUMENT)

        # parameterList
        self.compile_parameter_list()

        # )
        self._eat(RIGHT_PAREN)

        # <subroutineBody>
        # {
        self._eat(LEFT_BRACE)
        while self.current_token_value == VAR:  # order matters, simplify
            self.compile_var_dec()
        n_vars = self.symbol_table.var_count(LOCAL)
        self.vm_stream.write_function(f'{self.current_class_name}.{subroutine_name}', n_vars)

        # special handling of the constructor
        if subroutine_type == CONSTRUCTOR:
            # initialization
            n_fields = self.symbol_table.var_count(FIELD)
            self.vm_stream.write_push("constant", n_fields)
            self.vm_stream.write_call("Memory.alloc", n_args=1)
            self.vm_stream.write_pop("pointer", 0)
        elif subroutine_type == METHOD:  # THIS = argument 0
            self.vm_stream.write_push("argument", 0)
            self.vm_stream.write_pop("pointer", 0)

        self.compile_statements()
        # }
        self._eat(RIGHT_BRACE)
        # </subroutineBody>

        # </subroutineDec>

    def compile_parameter_list(self):
        """compile a jack parameter list which is 0 or more list
        """
        # <parameterList>

        # ((type varName) (, type varName)*)?
        while True:
            _type = self.current_token_value
            if _type in {INT, CHAR, BOOLEAN}:
                self._eat(_type)
            elif self.current_token_type == IDENTIFIER:
                self._eat(IDENTIFIER)
            else:
                break
            name = self.current_token_value
            self.symbol_table.define(name=name, _type=_type, kind=ARGUMENT)
            self._eat(IDENTIFIER)
            if not self.current_token_value == COMMA:
                break
            self._eat(COMMA)

    # def compile_subroutine_body(self):
    #     """compile a jack subroutine body which is varDec* statements
    #     """
    #     # <subroutineBody>
    #     # {
    #     self._eat(LEFT_BRACE)
    #     while self.current_token_value == VAR:  # order matters, simplify
    #         self.compile_var_dec()
    #     self.compile_statements()
    #     # }
    #     self._eat(RIGHT_BRACE)
    #     # </subroutineBody>

    def compile_var_dec(self):
        """compile jack variable declaration, varDec*, only called if current token is var
        add the variable to symbol table
        """
        # <varDec>
        # VAR
        self._eat(VAR)

        # type varName (',' varName)*;
        for _type, name in self._handle_type_var_name():
            self.symbol_table.define(name=name, _type=_type, kind=LOCAL)
        # </varDec>

    def compile_statements(self):
        """
        match the current token value to the matching jack statement
        """
        # <statements>

        while self.current_token_value in {LET, IF, WHILE, DO, RETURN}:
            {
                LET: self.compile_let_statement,
                IF: self.compile_if_statement,
                WHILE: self.compile_while_statement,
                DO: self.compile_do_statement,
                RETURN: self.compile_return_statement,
            }[self.current_token_value]()

        # </statements>

    def compile_let_statement(self):
        """
        compile jack let statement
        """
        # <letStatement>

        # let
        self._eat(LET)

        # varName
        name = self.current_token_value
        self._eat(IDENTIFIER)

        # ( '[' expression ']')? - a bit involved to avoid arr1[exp1] = exp2 where exp2 might be arr2[exp3] writing
        # to THAT at same time
        if self.current_token_value == LEFT_BRACKET:
            if self.symbol_table.contains(name):
                kind = self.symbol_table.kind_of(name)
                kind = THIS if kind == FIELD else kind
                index = self.symbol_table.index_of(name)
                self.vm_stream.write_push(kind, index)  # arr1
            self._eat(LEFT_BRACKET)
            self.compile_expression()  # exp1
            self.vm_stream.write_arithmetic("add")
            self._eat(RIGHT_BRACKET)
            self._eat(EQUAL_SIGN)
            self.compile_expression()  # exp2
            self.vm_stream.write_pop("temp", 0)
            self.vm_stream.write_pop("pointer", 1)
            self.vm_stream.write_push("temp", 0)
            self.vm_stream.write_pop(THAT, 0)
        else:
            # =
            self._eat(EQUAL_SIGN)
            # expression
            self.compile_expression()

            if self.symbol_table.contains(name):
                kind = self.symbol_table.kind_of(name)
                kind = THIS if kind == FIELD else kind
                index = self.symbol_table.index_of(name)
                self.vm_stream.write_pop(kind, index)
            else:
                raise CompileException(f"{name} is not defined")
        # ;
        self._eat(SEMI_COLON)

        # <letStatement>

    def compile_if_statement(self):
        """
        compile jack if statement
        """

        label_if_true = self.get_next_label(IF_TRUE)
        label_if_false = self.get_next_label(IF_FALSE)

        # <ifStatement>
        # if
        self._eat(IF)

        # (expression)
        self._eat(LEFT_PAREN)
        self.compile_expression()
        self._eat(RIGHT_PAREN)

        self.vm_stream.write_arithmetic("not")
        self.vm_stream.write_if_goto(label_if_false)
        # {statements1}
        self._handle_statements_within_braces()

        self.vm_stream.write_goto(label_if_true)

        self.vm_stream.write_label(label_if_false)
        if self.current_token_value == ELSE:
            self._eat(ELSE)
            # {statements2}
            self._handle_statements_within_braces()
        self.vm_stream.write_label(label_if_true)

        # <ifStatement>/

    def _handle_statements_within_braces(self):
        # {
        self._eat(LEFT_BRACE)
        # statements
        while self.current_token_value in {LET, IF, WHILE, DO, RETURN}:
            self.compile_statements()
        # }
        self._eat(RIGHT_BRACE)

    def compile_while_statement(self):
        """
        compile jack while statement
        """

        # <whileStatement>

        # while
        self._eat(WHILE)

        label_while_exp = self.get_next_label(WHILE_EXP)
        label_while_end = self.get_next_label(WHILE_END)

        self.vm_stream.write_label(label_while_exp)

        # (expression)
        self._eat(LEFT_PAREN)
        self.compile_expression()
        self._eat(RIGHT_PAREN)

        self.vm_stream.write_arithmetic("not")
        self.vm_stream.write_if_goto(label_while_end)

        # {statements}
        self._handle_statements_within_braces()
        self.vm_stream.write_goto(label_while_exp)

        self.vm_stream.write_label(label_while_end)

        # <whileStatement>

    def compile_do_statement(self):
        """
        compile jack do statement
        """

        # <doStatement>
        # do
        self._eat(DO)
        # subroutineName | (className | varName)'.'subroutineName
        first_token = self.current_token_value
        self._eat(IDENTIFIER)
        look_ahead_token = self.current_token_value
        self._compile_subroutine_call(first_token, look_ahead_token)
        self.vm_stream.write_pop("temp", 0)
        # ;
        self._eat(SEMI_COLON)
        # </doStatement>

    def _compile_subroutine_call(self, first_token, look_ahead_token):
        # TODO: refactor later
        # subroutineName | (className | varName)'.'subroutineName
        n_args = 0
        if look_ahead_token != DOT:  # bar()
            class_name = self.current_class_name
            subroutine_name = first_token
            self.vm_stream.write_push("pointer", 0)  # pushing current object as 1st arg
            n_args += 1
        elif first_token.lower() == first_token:  # obj.bar()
            self._eat(DOT)
            subroutine_name = self.current_token_value
            self._eat(IDENTIFIER)
            if self.symbol_table.contains(first_token):
                class_name = self.symbol_table.type_of(first_token)
                kind = self.symbol_table.kind_of(first_token)
                kind = THIS if kind == FIELD else kind
                index = self.symbol_table.index_of(first_token)
                self.vm_stream.write_push(kind, index)
            else:
                raise CompileException(f"{first_token} is not defined")
            n_args += 1
        else:  # Foo.bar()
            class_name = first_token
            self._eat(DOT)
            subroutine_name = self.current_token_value
            self._eat(IDENTIFIER)

        subroutine_full_name = f'{class_name}.{subroutine_name}'

        # (expressionList)
        self._eat(LEFT_PAREN)

        n_args += self.compile_expression_list()
        self.vm_stream.write_call(name=subroutine_full_name, n_args=n_args)

        self._eat(RIGHT_PAREN)

    def compile_return_statement(self):
        """
        compile jack return statement
        """
        # <returnStatement>
        # return
        self._eat(RETURN)

        # expression?
        if self.current_token_value == SEMI_COLON:
            # every func in jack must return something (we use 0 as dummy return)
            self.vm_stream.write_push("constant", 0)
        else:
            self.compile_expression()

        self.vm_stream.write_return()

        # ;
        self._eat(SEMI_COLON)

        # </returnStatement>

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
        # term
        self.compile_term()

        # (op term)*
        while self.current_token_value in OP:
            operation = VM_OPERATIONS.get(self.current_token_value, self.current_token_value)
            self._eat(self.current_token_value)
            self.compile_term()
            self.vm_stream.write_arithmetic(operation)

        # </expression>

    def compile_term(self):
        """
        compile jack term
        """

        # <term>
        current_token_value, current_token_type = self.current_token_value, self.current_token_type
        if current_token_type == INT_CONSTANT:
            self.vm_stream.write_push("constant", current_token_value)
            self._eat(INT_CONSTANT)
        elif current_token_type == STR_CONSTANT:
            current_value = current_token_value.strip('"')
            self.vm_stream.write_push("constant", len(current_value))
            self.vm_stream.write_call("String.new", 1)
            for c in current_value:
                self.vm_stream.write_push("constant", ord(c))
                self.vm_stream.write_call("String.appendChar", 2)
            self._eat(STR_CONSTANT)
        elif current_token_value in KEYWORD_CONSTANT:
            if current_token_value == NULL or current_token_value == FALSE:
                self.vm_stream.write_push("constant", 0)
            elif current_token_value == TRUE:
                self.vm_stream.write_push("constant", 0)
                self.vm_stream.write_arithmetic("not")
            elif current_token_value == THIS:
                self.vm_stream.write_push("pointer", 0)
            self._eat(current_token_value)
        elif current_token_value in UNARY_OP:
            vm_command = "not" if current_token_value == TILDE else "neg"
            self._eat(current_token_value)
            self.compile_term()
            self.vm_stream.write_arithmetic(vm_command)
        elif current_token_value == LEFT_PAREN:  # '(' expression ')'
            self._eat(LEFT_PAREN)
            self.compile_expression()
            self._eat(RIGHT_PAREN)
        else:  # identifier
            first_token = self.current_token_value
            self._eat(IDENTIFIER)
            look_ahead_token = self.current_token_value

            # subroutineName | (className | varName)'.'subroutineName
            if look_ahead_token == LEFT_PAREN or look_ahead_token == DOT:
                self._compile_subroutine_call(first_token, look_ahead_token)
            # varName'[' expression ']'
            elif look_ahead_token == LEFT_BRACKET:
                if self.symbol_table.contains(first_token):
                    kind = self.symbol_table.kind_of(first_token)
                    kind = THIS if kind == FIELD else kind
                    index = self.symbol_table.index_of(first_token)
                    self.vm_stream.write_push(kind, index)
                else:
                    raise CompileException(f"{first_token} is not defined")

                self._eat(LEFT_BRACKET)
                self.compile_expression()
                self.vm_stream.write_arithmetic("add")
                self.vm_stream.write_pop("pointer", 1)
                self.vm_stream.write_push(THAT, 0)
                self._eat(RIGHT_BRACKET)
            # foo
            else:
                if self.symbol_table.contains(first_token):
                    kind = self.symbol_table.kind_of(first_token)
                    kind = THIS if kind == FIELD else kind
                    index = self.symbol_table.index_of(first_token)
                    self.vm_stream.write_push(kind, index)
                else:
                    raise CompileException(f"{first_token} is not defined")
        # </term>

    def compile_expression_list(self):
        """
        compile jack expression list and return number of expressions compiled
        """
        # <expressionList>
        # (expression (',' expression)*)?
        n_expressions = 0
        if not self.current_token_value == RIGHT_PAREN:
            while True:
                self.compile_expression()
                n_expressions += 1
                if not self.current_token_value == COMMA:
                    break
                self._eat(COMMA)

        # </expressionList>
        return n_expressions


# TODO: implement
# Module 3: Symbol Table, we need 2: class-level and subroutine-level
""" Symbol tables
name    type    kind        #
x       int     field       0
y       int     field       1
"""


# # helper enums
# class JType(enum.Enum):
#     JInt = 1
#     JChar = 2
#     JBoolean = 3
#     JCustom = 4  # class name
#

# class JVarKind(enum.Enum):
#     field = 1
#     static = 2
#     local = 3
#     argument = 4
# def __str__(self):
#     return self.name
#
# def __repr__(self):
#     return self.name

# class JVarScope(enum.Enum):
#     class_level = 1
#     subroutine_level = 2


class JVariable(NamedTuple):
    name: str
    type: str
    kind: str
    index: int

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, JVariable):
            return False
        return self.name == o.name and self.type == o.type and self.kind == o.kind


class SymbolTable:
    def __init__(self):
        self._class_level_data = []
        self._local_index = self._argument_index = self._field_index = self._static_index = 0
        self._sub_level_data = []

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        result = ''
        if self._class_level_data:
            result += '\nclass level:\n' + '\n'.join(map(str, self._class_level_data))
        if self._sub_level_data:
            result += '\n\nsub level:\n' + '\n'.join(map(str, self._sub_level_data)) + '\n'
        return result

    # starts a new subroutine scope
    def start_subroutine(self):
        self._sub_level_data = []
        self._local_index = self._argument_index = 0

    # defines a new identifier of given parameters and assign it a running index
    def define(self, name: str, _type: str, kind: Optional[str]):
        for arg in name, _type, kind:
            if not arg:
                raise CompileException(f"'{arg}' is not a valid variable name")

        if kind == LOCAL:
            j_var = JVariable(name, _type, kind, self._local_index)
            self._local_index += 1
            self._sub_level_data.append(j_var)
        elif kind == ARGUMENT:
            j_var = JVariable(name, _type, kind, self._argument_index)
            self._argument_index += 1
            self._sub_level_data.append(j_var)
        elif kind == FIELD:
            j_var = JVariable(name, _type, kind, self._field_index)
            self._field_index += 1
            self._class_level_data.append(j_var)
        elif kind == STATIC:
            j_var = JVariable(name, _type, kind, self._static_index)
            self._static_index += 1
            self._class_level_data.append(j_var)

    def var_count(self, kind: str):
        """
        return the number of variables of the given kind
        :param kind: kind of jack variable
        :return: number of variables of kind in symbol table
        """
        if kind in (FIELD, STATIC):
            return sum(kind == var.kind for var in self._class_level_data)
        if kind in (LOCAL, ARGUMENT):
            return sum(kind == var.kind for var in self._sub_level_data)
        raise CompileException(f"{kind} is not supported!")

    def kind_of(self, var: str):
        """
        return the jack kind of the identifier var
        :param var: the identifier or variable to look for
        :return: kind of var
        """
        kind = self._get_var_property(var, 'kind')
        if kind is not None:
            return kind
        raise CompileException(f"{var} is not defined!!")

    def type_of(self, var: str):
        """
        return the jack type of the identifier var
        :param var: the identifier or variable to look for
        :return: type of var
        """
        _type = self._get_var_property(var, 'type')
        if _type is not None:
            return _type
        raise CompileException(f"{var} is not defined!")

    def index_of(self, var: str):
        """
        return the index of the identifier var
        :param var: the identifier or variable to look for
        :return:
        """
        index = self._get_var_property(var, 'index')
        if index is not None:
            return index
        return -1

    def contains(self, var: str):
        if self.index_of(var) == -1:
            return False
        return True

    def _get_var_property(self, var: str, _property: str):
        property_getter = attrgetter(_property)
        for sub_var in self._sub_level_data:
            if var == sub_var.name:
                return property_getter(sub_var)
        for class_var in self._class_level_data:
            if var == class_var.name:
                return property_getter(class_var)
        return None


# Module 1: Jack compiler (ui)
def handle_file(src_path):
    logging.info(f'compiling {src_path}')
    out_vm_path = src_path.replace(SRC_FILE_EXT, VM_FILE_EXT)
    with open(src_path) as src_stream, \
            open(out_vm_path, 'w') as vm_stream:
        compilation_engine = CompilationEngine(JackTokenizer(src_stream), VMWriter(vm_stream))
        compilation_engine.compile_class()
    logging.info(f'generated {out_vm_path}')


def handle_dir(path):
    for f in os.listdir(path):
        if f.endswith(SRC_FILE_EXT):
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
