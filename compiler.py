import logging
import os
import re
import sys
from operator import attrgetter
from typing import NamedTuple, TextIO, Optional

from constants import *


class ParseException(Exception):
    pass


class CompileException(Exception):
    pass


class Token(NamedTuple):
    type_: str
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
        IDENTIFIER: r'[a-zA-Z_][a-zA-Z0-9_]*',  # must be after keywords, in python re site, considered keyword as
        # part of ID pattern newline=r'\n',
        'mismatch': r'.',  # any other character
    }
    jack_token = re.compile(
        '|'.join(
            r'(?P<{}>{})'.format(token, specification)
            for token, specification in tokens_specifications.items()
        )
    )

    def __init__(self, in_stream):
        self.in_stream = in_stream
        self.src_base_name = os.path.split(self.in_stream.name)[-1].rpartition('.')[0]

    def start_tokenizer(self):
        line_number = 1
        for m in self.jack_token.finditer(self.in_stream.read()):
            token_type = m.lastgroup
            if token_type is None:
                raise ParseException('token type cannot be None')
            token_value = m.group(token_type)
            match token_type:
                case 'integerConstant':
                    token_value = token_value
                case 'identifier':
                    if token_value in self.KEYWORDS:
                        token_type = KEYWORD
                case 'newline':
                    line_number += 1
                    continue
                case 'comment':
                    line_number += token_value.count('\n')
                    continue
                case 'space':
                    continue
                case 'mismatch':
                    raise ParseException(
                        f'got wrong jack token: {token_value} in line {line_number}')
            yield Token(token_type, token_value, line_number)


# Module 4: VMWriter, generates VM code
class Writer:
    def __init__(self, out_stream: TextIO = None):
        if out_stream is not None:
            self.out_stream = out_stream
        else:
            self.out_stream = sys.stdout


# TODO: Can we implement visitor pattern for this (python cookbook 8.21)
class VMWriter(Writer):
    def _write_line(self, text):
        self.out_stream.write(text + NEWLINE)

    def write_push(self, segment, index):
        self._write_line(f"push {segment} {index}")

    def write_pop(self, segment, index):
        self._write_line(f"pop {segment} {index}")

    def write_arithmetic(self, vm_command):
        self._write_line(f"{vm_command}")

    # writes a vm label command
    def write_label(self, label):
        self._write_line(f"label {label}")

    # writes a vm goto command
    def write_goto(self, label):
        self._write_line(f"goto {label}")

    # writes a vm if-goto command
    def write_if_goto(self, label):
        self._write_line(f"if-goto {label}")

    # writes a vm call command
    def write_call(self, name: str, n_args: int):
        self._write_line(f"call {name} {n_args}")

    # write a vm function
    def write_function(self, name: str, n_locals: int):
        self._write_line(f"function {name} {n_locals}")

    # writes a vm return command
    def write_return(self):
        self._write_line(f"return")

    # closes the output file, I guess not needed here, it is java-style
    def close(self):
        pass


# TODO: Can we implement visitor pattern for this (python cookbook 8.21)
# Module 5: CompilationEngine
class CompilationEngine:
    def __init__(self, jack_tokenizer: JackTokenizer, vm_stream: VMWriter):
        """ initialize the compilation engine which parses tokens from tokensStream and write in vm_stream
        INVARIANT: current_token is the token we are handling now given eat() is last to run in handling it
        Args:
            jack_tokenizer (JackTokenizer): the jack source code tokenizer
            vm_stream (VMWriter): writer of compiled vm code
        """
        self.jack_tokenizer = jack_tokenizer
        self.tokens_stream = jack_tokenizer.start_tokenizer()
        self.vm_stream = vm_stream
        self.symbol_table = SymbolTable()
        self.current_token = Token('', '', -1)  # instead of none for easier type checking

        # labels indices
        self.labels_indices = {}

    @property
    def current_class_name(self):
        return self.jack_tokenizer.src_base_name

    def get_next_label(self, label_prefix):
        index = self.labels_indices.setdefault(label_prefix, -1) + 1
        self.labels_indices[label_prefix] = index
        return f'{label_prefix}{index}'

    def eat(self, s):
        """advance to next token if given string is same as the current token, otherwise raise error
        Args:
            s (str): string to match current token against
        Raises:
            ParseException: in case no match
        """
        if s == self.current_token.value or \
                (s == self.current_token.type_ and s in {INT_CONSTANT, STR_CONSTANT, IDENTIFIER}):
            try:
                self.current_token = next(self.tokens_stream)
            except StopIteration:
                if s != RIGHT_BRACE:  # last token
                    raise ParseException(f'Error, reached end of file\n{str(self.current_token)}')
        else:
            raise ParseException(
                f'Got wrong token in line {self.current_token.line_number}: '
                f'{self.current_token.value}, expected: {s!r}\n{str(self.current_token)}')

    def compile_class(self):
        """Starting point in compiling a jack source file
        """
        # first token
        try:
            self.current_token = next(self.tokens_stream)
        except StopIteration:  # jack source file is empty
            return

        # <class>
        # class
        self.eat(CLASS)

        # className
        try:
            assert self.current_token.value == self.current_class_name
        except AssertionError:
            raise CompileException(
                f"class {self.current_token.value} does not match filename {self.current_class_name}")
        self.eat(IDENTIFIER)

        # {
        self.eat(LEFT_BRACE)

        # classVarDec*
        while self.current_token.value in {STATIC, FIELD}:
            self.compile_class_var_dec()

        # subroutineDec*
        while self.current_token.value in {CONSTRUCTOR, FUNCTION, METHOD}:
            self.compile_subroutine_dec()

        # }
        self.eat(RIGHT_BRACE)

        # </class>

    def compile_class_var_dec(self):
        """compile a jack class variable declarations
        ASSUME: only called if current token's value is static or field
        """
        # <classVarDec>
        # field | static
        kind = self.current_token.value
        self.eat(self.current_token.value)

        # type varName (',' varName)*;
        _type = self._eat_and_return_var_type()
        comma_before = False
        while self.current_token.value != SEMI_COLON:
            if comma_before: self.eat(COMMA)
            name = self._eat_and_return_var_name()
            comma_before = True
            self.symbol_table.define(name=name, _type=_type, kind=kind)

        self.eat(SEMI_COLON)

        # varName
        # </classVarDec>

    def _eat_and_return_var_type(self):
        # type
        if self.current_token.value in {INT, CHAR, BOOLEAN}:
            _type = self.current_token.value
            self.eat(self.current_token.value)
        elif self.current_token.type_ == IDENTIFIER:
            _type = self.current_token.value
            self.eat(IDENTIFIER)
        else:
            raise CompileException(
                f'Unidentified variable type: \'{self.current_token.value}\' in line {self.current_token.line_number}')
        return _type

    def _eat_and_return_var_name(self):
        name = self.current_token.value
        self.eat(IDENTIFIER)
        return name

    def compile_subroutine_dec(self):
        """compile a jack class subroutine declarations
        ASSUME: only called if current token's value is constructor, function or method
        """
        self.symbol_table.start_subroutine()
        # <subroutineDec>
        # constructor | function | method
        subroutine_type = self.current_token.value
        if subroutine_type in (CONSTRUCTOR, FUNCTION, METHOD):
            self.eat(subroutine_type)

        # builtin type | className
        if self.current_token.value in (VOID, INT, CHAR, BOOLEAN):
            self.eat(self.current_token.value)
        elif self.current_token.type_ == IDENTIFIER:
            self.eat(IDENTIFIER)

        # subroutineName
        subroutine_name = self.current_token.value
        self.eat(IDENTIFIER)

        # (
        self.eat(LEFT_PAREN)

        # add this as argument in case the subroutine_type is a method
        if subroutine_type == METHOD:
            self.symbol_table.define(name=THIS, _type=self.current_class_name, kind=ARGUMENT)

        # parameterList
        self.compile_parameter_list()

        # )
        self.eat(RIGHT_PAREN)

        # <subroutineBody>
        # {
        self.eat(LEFT_BRACE)
        while self.current_token.value == VAR:  # order matters, simplify
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
        self.eat(RIGHT_BRACE)
        # </subroutineBody>

        # </subroutineDec>

    def compile_parameter_list(self):
        """compile a jack parameter list which is 0 or more list
        """
        # <parameterList>

        comma_before = False
        while self.current_token.value != RIGHT_PAREN:
            if comma_before: self.eat(COMMA)
            _type = self._eat_and_return_var_type()
            name = self._eat_and_return_var_name()
            self.symbol_table.define(name=name, _type=_type, kind=ARGUMENT)
            comma_before = True

    def compile_var_dec(self):
        """compile jack variable declaration, varDec*, only called if current token is var
        add the variable to symbol table
        """
        # <varDec>
        # VAR
        self.eat(VAR)

        # type varName (',' varName)*;
        _type = self._eat_and_return_var_type()
        comma_before = False
        while self.current_token.value != SEMI_COLON:
            if comma_before: self.eat(COMMA)
            name = self._eat_and_return_var_name()
            self.symbol_table.define(name=name, _type=_type, kind=LOCAL)
            comma_before = True

        self.eat(SEMI_COLON)
        # </varDec>

    def compile_statements(self):
        """
        match the current token value to the matching jack statement
        """
        # <statements>

        while self.current_token.value in {LET, IF, WHILE, DO, RETURN}:
            {
                LET: self.compile_let_statement,
                IF: self.compile_if_statement,
                WHILE: self.compile_while_statement,
                DO: self.compile_do_statement,
                RETURN: self.compile_return_statement,
            }[self.current_token.value]()

        # </statements>

    def compile_let_statement(self):
        """
        compile jack let statement
        """
        # <letStatement>

        # let
        self.eat(LET)

        # varName
        name = self.current_token.value
        self.eat(IDENTIFIER)

        # ( '[' expression ']')? - a bit involved to avoid arr1[exp1] = exp2 where exp2 might be arr2[exp3] writing
        # to THAT at same time
        if self.current_token.value == LEFT_BRACKET:
            segment = SEGMENT_OF_KIND.get(self.symbol_table.kind_of(name))
            index = self.symbol_table.index_of(name)
            self.vm_stream.write_push(segment, index)  # arr1
            self.eat(LEFT_BRACKET)
            self.compile_expression()  # exp1
            self.vm_stream.write_arithmetic("add")
            self.eat(RIGHT_BRACKET)
            self.eat(EQUAL_SIGN)
            self.compile_expression()  # exp2
            self.vm_stream.write_pop("temp", 0)
            self.vm_stream.write_pop("pointer", 1)
            self.vm_stream.write_push("temp", 0)
            self.vm_stream.write_pop(THAT, 0)
        else:
            # =
            self.eat(EQUAL_SIGN)
            # expression
            self.compile_expression()
            segment = SEGMENT_OF_KIND.get(self.symbol_table.kind_of(name))
            index = self.symbol_table.index_of(name)
        self.vm_stream.write_pop(segment, index)
        # ;
        self.eat(SEMI_COLON)

        # <letStatement>

    def compile_if_statement(self):
        """
        compile jack if statement
        """

        label_if_true = self.get_next_label(IF_TRUE)
        label_if_false = self.get_next_label(IF_FALSE)

        # <ifStatement>
        # if
        self.eat(IF)

        # (expression)
        self.eat(LEFT_PAREN)
        self.compile_expression()
        self.eat(RIGHT_PAREN)

        self.vm_stream.write_arithmetic("not")
        self.vm_stream.write_if_goto(label_if_false)
        # {statements1}
        self._handle_statements_within_braces()

        self.vm_stream.write_goto(label_if_true)

        self.vm_stream.write_label(label_if_false)
        if self.current_token.value == ELSE:
            self.eat(ELSE)
            # {statements2}
            self._handle_statements_within_braces()
        self.vm_stream.write_label(label_if_true)

        # <ifStatement>/

    def _handle_statements_within_braces(self):
        # {
        self.eat(LEFT_BRACE)
        # statements
        while self.current_token.value in {LET, IF, WHILE, DO, RETURN}:
            self.compile_statements()
        # }
        self.eat(RIGHT_BRACE)

    def compile_while_statement(self):
        """
        compile jack while statement
        """

        # <whileStatement>

        # while
        self.eat(WHILE)

        label_while_exp = self.get_next_label(WHILE_EXP)
        label_while_end = self.get_next_label(WHILE_END)

        self.vm_stream.write_label(label_while_exp)

        # (expression)
        self.eat(LEFT_PAREN)
        self.compile_expression()
        self.eat(RIGHT_PAREN)

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
        self.eat(DO)
        # subroutineName | (className | varName)'.'subroutineName
        first_token = self.current_token.value
        self.eat(IDENTIFIER)
        look_ahead_token = self.current_token.value
        self._compile_subroutine_call(first_token, look_ahead_token)
        self.vm_stream.write_pop("temp", 0)
        # ;
        self.eat(SEMI_COLON)
        # </doStatement>

    def _compile_subroutine_call(self, first_token, look_ahead_token):
        # subroutineName | (className | varName)'.'subroutineName
        n_args = 0
        if look_ahead_token != DOT:  # bar()
            class_name = self.current_class_name
            subroutine_name = first_token
            self.vm_stream.write_push("pointer", 0)  # pushing current object as 1st arg
            n_args += 1
        elif first_token.lower() == first_token:  # obj.bar()
            self.eat(DOT)
            subroutine_name = self.current_token.value
            self.eat(IDENTIFIER)
            class_name = self.symbol_table.type_of(first_token)
            segment = SEGMENT_OF_KIND.get(self.symbol_table.kind_of(first_token))
            index = self.symbol_table.index_of(first_token)
            self.vm_stream.write_push(segment, index)
            n_args += 1
        else:  # Foo.bar()
            class_name = first_token
            self.eat(DOT)
            subroutine_name = self.current_token.value
            self.eat(IDENTIFIER)

        subroutine_full_name = f'{class_name}.{subroutine_name}'

        # (expressionList)
        self.eat(LEFT_PAREN)

        n_args += self.compile_expression_list()
        self.vm_stream.write_call(name=subroutine_full_name, n_args=n_args)

        self.eat(RIGHT_PAREN)

    def compile_return_statement(self):
        """
        compile jack return statement
        """
        # <returnStatement>
        # return
        self.eat(RETURN)

        # expression?
        if self.current_token.value == SEMI_COLON:
            # every func in jack must return something (we use 0 as dummy return)
            self.vm_stream.write_push("constant", 0)
        else:
            self.compile_expression()

        self.vm_stream.write_return()

        # ;
        self.eat(SEMI_COLON)

        # </returnStatement>

    def compile_expression(self):
        """
        compile jack expression
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
        while self.current_token.value in OP:
            operation = VM_OPERATIONS.get(self.current_token.value, self.current_token.value)
            self.eat(self.current_token.value)
            self.compile_term()
            self.vm_stream.write_arithmetic(operation)

        # </expression>

    def compile_term(self):
        """
        compile jack term
        """
        # <term>
        first_token = self.current_token.value
        if self.current_token.type_ == INT_CONSTANT:
            self._handle_term_integer_constant()
        elif self.current_token.type_ == STR_CONSTANT:
            self._handle_term_string_constant()
        elif first_token in TRUE_FALSE_NULL_THIS:
            self._handle_term_keyword_constant()
        elif first_token in UNARY_OP:
            self._handle_term_unary_operator()
        elif first_token == LEFT_PAREN:  # '(' expression ')'
            self.eat(LEFT_PAREN)
            self.compile_expression()
            self.eat(RIGHT_PAREN)
        else:  # identifier
            self._handle_term_identifier()

    def _handle_term_identifier(self):
        first_token = self.current_token.value
        self.eat(IDENTIFIER)
        look_ahead_token = self.current_token.value

        # subroutineName | (className | varName)'.'subroutineName
        if look_ahead_token == LEFT_PAREN or look_ahead_token == DOT:
            self._compile_subroutine_call(first_token, look_ahead_token)
        else:  # foo
            segment = SEGMENT_OF_KIND.get(self.symbol_table.kind_of(first_token))
            index = self.symbol_table.index_of(first_token)
            self.vm_stream.write_push(segment, index)
            # foo'[' expression ']'
            if look_ahead_token == LEFT_BRACKET:
                self.eat(LEFT_BRACKET)
                self.compile_expression()
                self.vm_stream.write_arithmetic("add")
                self.vm_stream.write_pop("pointer", 1)
                self.vm_stream.write_push(THAT, 0)
                self.eat(RIGHT_BRACKET)
        # </term>

    def _handle_term_integer_constant(self):
        self.vm_stream.write_push("constant", int(self.current_token.value))
        self.eat(INT_CONSTANT)

    def _handle_term_string_constant(self):
        current_value = self.current_token.value.strip('"')
        self.vm_stream.write_push("constant", len(current_value))
        self.vm_stream.write_call("String.new", 1)
        for c in current_value:
            self.vm_stream.write_push("constant", ord(c))
            self.vm_stream.write_call("String.appendChar", 2)
        self.eat(STR_CONSTANT)

    def _handle_term_keyword_constant(self):
        token = self.current_token.value
        if token == NULL or token == FALSE:
            self.vm_stream.write_push("constant", 0)
        elif token == TRUE:
            self.vm_stream.write_push("constant", 0)
            self.vm_stream.write_arithmetic("not")
        elif token == THIS:
            self.vm_stream.write_push("pointer", 0)
        self.eat(token)

    def _handle_term_unary_operator(self):
        token = self.current_token.value
        vm_command = "not" if token == TILDE else "neg"
        self.eat(token)
        self.compile_term()
        self.vm_stream.write_arithmetic(vm_command)

    def compile_expression_list(self):
        """
        compile jack expression list and return number of expressions compiled
        """
        # <expressionList>
        # (expression (',' expression)*)?
        n_expressions = 0
        if not self.current_token.value == RIGHT_PAREN:
            while True:
                self.compile_expression()
                n_expressions += 1
                if not self.current_token.value == COMMA:
                    break
                self.eat(COMMA)

        # </expressionList>
        return n_expressions


# Module 3: Symbol Table, we need 2: class-level and subroutine-level
""" Symbol tables
name    type    kind        #
x       int     field       0
y       int     field       1
"""


class JVariable(NamedTuple):
    name: str
    type: str
    kind: str
    index: int

    # def __eq__(self, o: object) -> bool:
    #     if not isinstance(o, JVariable):
    #         return False
    #     return self.name == o.name and self.type == o.type and self.kind == o.kind


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
    # TODO: the check needs revision
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
        if kind == FIELD:
            return self._field_index
        if kind == STATIC:
            return self._static_index
        if kind == LOCAL:
            return self._local_index
        if kind == ARGUMENT:
            return self._argument_index
        raise CompileException(f"{kind} is not supported!")

    def kind_of(self, var: str):
        """
        return the jack kind of the identifier var
        :param var: the identifier or variable to look for
        :return: kind of var or THIS (if kind is FIELD)
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
        raise CompileException(f"{var} is not defined!")

    # def contains(self, var: str):
    #     if self.index_of(var) == -1:
    #         return False
    #     return True

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
