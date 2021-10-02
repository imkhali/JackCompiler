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
XML_FILE_EXT = '_test.xml'
VM_FILE_EXT = '_test.vm'
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


class XMLWriter(Writer):
    special_xml = {
        LESS_THAN: '&lt;',
        GREATER_THAN: '&gt;',
        AMPERSAND: '&amp;',
        DOUBLE_QUOTES: '&quot;'
    }

    def __init__(self, out_stream):
        super().__init__(out_stream)
        self.indent_level = 0

    def reindent(self, indent=1):
        self.indent_level += indent

    def deindent(self, indent=1):
        self.indent_level -= indent

    def write_tag_value(self, name=None, value=None, tags=None):
        """writes xml tagged jack token to xmlFileStream
        Args:
            name (str): type of token
            value (str | integer): value of token
            tags: name, value mapping
        """
        if not tags:
            tags = {name: value}
        for name, value in tags.items():
            value = self.special_xml.get(value, value)
            indent = ' ' * INDENT_NUM_SPACES * self.indent_level
            self.out_stream.write(f'{indent}<{name}> {value} </{name}>{NEWLINE}')

    def write_open_tag(self, tag):
        """writes xml open tag with given tag
        Args:
            tag (str): xml tag
        """
        indent = ' ' * INDENT_NUM_SPACES * self.indent_level
        self.out_stream.write(f'{indent}<{tag}>{NEWLINE}')

    def write_close_tag(self, tag):
        """writes xml close tag with given tag
        Args:
            tag (str): xml tag
        """
        indent = ' ' * INDENT_NUM_SPACES * self.indent_level
        self.out_stream.write(f'{indent}</{tag}>{NEWLINE}')


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


# TODO: change to output the vm code (no more xml)
# Module 5: CompilationEngine
class CompilationEngine:
    def __init__(self, jack_tokenizer: JackTokenizer, xml_stream: XMLWriter, vm_stream: VMWriter):
        """ initialize the compilation engine which parses tokens from tokensStream and write in xmlFileStream
        INVARIANT: current_token is the token we are handling now given _eat() is last to run in handling it
        Args:
            jack_tokenizer (JackTokenizer): the jack source code tokenizer
            xml_stream (XMLWriter): writer of parsed xml code
            vm_stream (VMWriter): writer of compiled vm code
        """
        self.jack_tokenizer = jack_tokenizer
        self.tokens_stream = jack_tokenizer.start_tokenizer()
        self.xml_stream = xml_stream
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
        self.xml_stream.write_open_tag(CLASS)
        self.xml_stream.reindent()

        # class
        self.xml_stream.write_tag_value(KEYWORD, self.current_token_value)
        self._eat(CLASS)

        # className
        self.xml_stream.write_open_tag(IDENTIFIER)
        self.xml_stream.reindent()
        self.xml_stream.write_tag_value(tags={
            "name": self.current_token_value,
            "kind": CLASS,
            "context": "declare"
        })
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag(IDENTIFIER)
        try:
            assert self.current_token_value == self.jack_tokenizer.src_base_name
        except AssertionError:
            raise CompileException(
                f"class {self.current_token_value} does not match filename {self.jack_tokenizer.src_base_name}")
        self._eat(IDENTIFIER)

        # {
        self.xml_stream.write_tag_value(SYMBOL, self.current_token_value)
        self._eat(LEFT_BRACE)

        # classVarDec*
        while self.current_token_value in {STATIC, FIELD}:
            self.compile_class_var_dec()

        # subroutineDec*
        while self.current_token_value in {CONSTRUCTOR, FUNCTION, METHOD}:
            self.symbol_table.start_subroutine()
            self.compile_subroutine_dec()

        # }
        self.xml_stream.write_tag_value(SYMBOL, self.current_token_value)
        self._eat(RIGHT_BRACE)

        # </class>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag(CLASS)

    def compile_class_var_dec(self):
        """compile a jack class variable declarations
        ASSUME: only called if current token's value is static or field
        """
        # <classVarDec>
        self.xml_stream.write_open_tag('classVarDec')
        self.xml_stream.reindent()

        # field | static
        self.xml_stream.write_tag_value(KEYWORD, self.current_token_value)
        kind = None
        if self.current_token_value in (STATIC, FIELD):
            kind = self.current_token_value
            self._eat(self.current_token_value)

        # varName
        for _type, name in self._handle_type_var_name():
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.symbol_table.define(name=name, _type=_type, kind=kind)
            self.xml_stream.write_tag_value(tags={
                "name": name,
                "kind": self.symbol_table.kind_of(name),
                "index": self.symbol_table.index_of(name),
                "context": "declare"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)

        # </classVarDec>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('classVarDec')

    def _handle_type_var_name(self):
        """
        :return: generator of jack variable (type, name)
        """
        # type
        _type = None
        if self.current_token_value in {INT, CHAR, BOOLEAN}:
            _type = self.current_token_value
            self.xml_stream.write_tag_value(KEYWORD, self.current_token_value)
            self._eat(self.current_token_value)
        elif self.current_token_type == IDENTIFIER:
            _type = self.current_token_value
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.xml_stream.write_tag_value(tags={
                "name": self.current_token_value,
                "kind": CLASS,
                "context": "use"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)
            # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)

        # varName (, varName)*;
        while True:
            name = self.current_token_value
            # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)
            yield _type, name
            if self.current_token_value == SEMI_COLON:
                break
            self.xml_stream.write_tag_value(SYMBOL, COMMA)
            self._eat(COMMA)
        self.xml_stream.write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

    def compile_subroutine_dec(self):
        """compile a jack class subroutine declarations
        ASSUME: only called if current token's value is constructor, function or method
        """
        # <subroutineDec>
        self.xml_stream.write_open_tag('subroutineDec')
        self.xml_stream.reindent()

        # constructor | function | method
        subroutine = self.current_token_value
        if subroutine in (CONSTRUCTOR, FUNCTION, METHOD):
            self.xml_stream.write_tag_value(KEYWORD, subroutine)
            self._eat(subroutine)

        # builtin type | className
        if self.current_token_value in (VOID, INT, CHAR, BOOLEAN):
            self.xml_stream.write_tag_value(KEYWORD, self.current_token_value)
            self._eat(self.current_token_value)
        elif self.current_token_type == IDENTIFIER:
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.xml_stream.write_tag_value(tags={
                "name": self.current_token_value,
                "kind": CLASS,
                "context": "use"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)
            # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)

        # subroutineName
        self.xml_stream.write_open_tag(IDENTIFIER)
        self.xml_stream.reindent()
        self.xml_stream.write_tag_value(tags={
            "name": self.current_token_value,
            "kind": SUBROUTINE,
            "context": "declare"
        })

        # TODO: How to get n_vars before compiling the subroutine body
        n_vars = -1
        self.vm_stream.write_function(f'{self.jack_tokenizer.src_base_name}.{self.current_token_value}', n_vars)

        self.xml_stream.deindent()
        self.xml_stream.write_close_tag(IDENTIFIER)
        # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
        self._eat(IDENTIFIER)

        # (
        self.xml_stream.write_tag_value(SYMBOL, self.current_token_value)
        self._eat(LEFT_PAREN)

        # add this as argument in case the subroutine is a constructor or a method
        if subroutine in (CONSTRUCTOR, METHOD):
            self.symbol_table.define(name=THIS, _type=self.jack_tokenizer.src_base_name, kind=ARGUMENT)
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.xml_stream.write_tag_value(tags={
                "name": THIS,
                "kind": self.symbol_table.kind_of(THIS),
                "index": self.symbol_table.index_of(THIS),
                "context": "declare"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)

        # parameterList
        self.compile_parameter_list()

        # )
        self.xml_stream.write_tag_value(SYMBOL, self.current_token_value)
        self._eat(RIGHT_PAREN)

        # subroutineBody
        self.compile_subroutine_body()

        # </subroutineDec>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('subroutineDec')

    def compile_parameter_list(self):
        """compile a jack parameter list which is 0 or more list
        """
        # <parameterList>
        self.xml_stream.write_open_tag('parameterList')
        self.xml_stream.reindent()

        # ((type varName) (, type varName)*)?
        while True:
            _type = self.current_token_value
            if _type in {INT, CHAR, BOOLEAN}:
                self.xml_stream.write_tag_value(KEYWORD, self.current_token_value)
                self._eat(_type)
            elif self.current_token_type == IDENTIFIER:
                self.xml_stream.write_open_tag(IDENTIFIER)
                self.xml_stream.reindent()
                self.xml_stream.write_tag_value(tags={
                    "name": _type,
                    "kind": CLASS,
                    "context": "use"
                })
                self.xml_stream.deindent()
                self.xml_stream.write_close_tag(IDENTIFIER)
                # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
                self._eat(IDENTIFIER)
            else:
                break
            name = self.current_token_value
            self.symbol_table.define(name=name, _type=_type, kind=ARGUMENT)
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.xml_stream.write_tag_value(tags={
                "name": name,
                "kind": self.symbol_table.kind_of(name),
                "index": self.symbol_table.index_of(name),
                "context": "declare"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)
            # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)
            if not self.current_token_value == COMMA:
                break
            self.xml_stream.write_tag_value(SYMBOL, COMMA)
            self._eat(COMMA)

        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('parameterList')

    def compile_subroutine_body(self):
        """compile a jack subroutine body which is varDec* statements
        """
        # <subroutineBody>
        self.xml_stream.write_open_tag('subroutineBody')
        self.xml_stream.reindent()

        # {
        self.xml_stream.write_tag_value(SYMBOL, LEFT_BRACE)
        self._eat(LEFT_BRACE)

        while self.current_token_value == VAR:  # order matters, simplify
            self.compile_var_dec()

        self.compile_statements()

        # }
        self.xml_stream.write_tag_value(SYMBOL, RIGHT_BRACE)
        self._eat(RIGHT_BRACE)

        # </subroutineBody>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('subroutineBody')

    def compile_var_dec(self):
        """compile jack variable declaration, varDec*, only called if current token is var
        add the variable to symbol table
        """
        # <varDec>
        self.xml_stream.write_open_tag('varDec')
        self.xml_stream.reindent()

        # VAR
        self.xml_stream.write_tag_value(KEYWORD, self.current_token_value)
        self._eat(VAR)

        # type varName (',' varName)*;
        for _type, name in self._handle_type_var_name():
            self.symbol_table.define(name=name, _type=_type, kind=LOCAL)
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.xml_stream.write_tag_value(tags={
                "name": name,
                "kind": self.symbol_table.kind_of(name),
                "index": self.symbol_table.index_of(name),
                "context": "declare"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)
        # </varDec>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('varDec')

    def compile_statements(self):
        """
        match the current token value to the matching jack statement
        """
        # <statements>
        self.xml_stream.write_open_tag('statements')
        self.xml_stream.reindent()

        while self.current_token_value in {LET, IF, WHILE, DO, RETURN}:
            {
                LET: self.compile_let_statement,
                IF: self.compile_if_statement,
                WHILE: self.compile_while_statement,
                DO: self.compile_do_statement,
                RETURN: self.compile_return_statement,
            }[self.current_token_value]()

        # </statements>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('statements')

    def compile_let_statement(self):
        """
        compile jack let statement
        """
        # <letStatement>
        self.xml_stream.write_open_tag('letStatement')
        self.xml_stream.reindent()

        # let
        self.xml_stream.write_tag_value(KEYWORD, LET)
        self._eat(LET)

        # varName
        name = self.current_token_value
        self.xml_stream.write_open_tag(IDENTIFIER)
        self.xml_stream.reindent()
        self.xml_stream.write_tag_value(tags={
            "name": name,
            "kind": self.symbol_table.kind_of(name),
            "index": self.symbol_table.index_of(name),
            "context": "assign"
        })
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag(IDENTIFIER)
        # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
        self._eat(IDENTIFIER)

        # ( '[' expression ']')?
        if self.current_token_value == LEFT_BRACKET:
            self.xml_stream.write_tag_value(SYMBOL, LEFT_BRACKET)
            self._eat(LEFT_BRACKET)
            self.compile_expression()
            self.xml_stream.write_tag_value(SYMBOL, RIGHT_BRACKET)
            self._eat(RIGHT_BRACKET)

        if self.symbol_table.contains(name):
            self._handle_lookup_var_xml(name)
            self.vm_stream.write_pop(
                self.symbol_table.kind_of(name),
                self.symbol_table.index_of(name)
            )
        else:
            raise CompileException(f"{name} is not defined")
        # =
        self.xml_stream.write_tag_value(SYMBOL, EQUAL_SIGN)
        self._eat(EQUAL_SIGN)

        # expression
        self.compile_expression()

        # ;
        self.xml_stream.write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

        # <letStatement>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('letStatement')

    def compile_if_statement(self):
        """
        compile jack if statement
        """

        # <ifStatement>
        self.xml_stream.write_open_tag('ifStatement')
        self.xml_stream.reindent()

        # if
        self.xml_stream.write_tag_value(KEYWORD, IF)
        self._eat(IF)

        # (expression)
        self.xml_stream.write_tag_value(SYMBOL, LEFT_PAREN)
        self._eat(LEFT_PAREN)
        self.compile_expression()
        self.xml_stream.write_tag_value(SYMBOL, RIGHT_PAREN)
        self._eat(RIGHT_PAREN)

        index = self.labels_indices.setdefault(IF_TRUE, -1) + 1
        label_if_true = f'{IF_TRUE}{index}'
        index = self.labels_indices.setdefault(IF_FALSE, -1) + 1
        label_if_false = f'{IF_FALSE}{index}'

        self.vm_stream.write_arithmetic("not")
        self.vm_stream.write_if_goto(label_if_false)

        self.vm_stream.write_goto(label_if_true)
        # {statements}
        self._handle_statements_within_braces()

        self.vm_stream.write_label(label_if_false)
        if self.current_token_value == ELSE:
            self.xml_stream.write_tag_value(KEYWORD, ELSE)
            self._eat(ELSE)
            self._handle_statements_within_braces()
        self.vm_stream.write_label(label_if_true)

        # <ifStatement>/
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('ifStatement')

    def _handle_statements_within_braces(self):
        # {
        self.xml_stream.write_tag_value(SYMBOL, LEFT_BRACE)
        self._eat(LEFT_BRACE)
        # statements
        while self.current_token_value in {LET, IF, WHILE, DO, RETURN}:
            self.compile_statements()
        # }
        self.xml_stream.write_tag_value(SYMBOL, RIGHT_BRACE)
        self._eat(RIGHT_BRACE)

    def compile_while_statement(self):
        """
        compile jack while statement
        """

        # <whileStatement>
        self.xml_stream.write_open_tag('whileStatement')
        self.xml_stream.reindent()

        # while
        self.xml_stream.write_tag_value(KEYWORD, WHILE)
        self._eat(WHILE)

        index = self.labels_indices.setdefault(WHILE_EXP, -1) + 1
        label_while_exp = f'{WHILE_EXP}{index}'
        index = self.labels_indices.setdefault(WHILE_END, -1) + 1
        label_while_end = f'{WHILE_END}{index}'

        self.vm_stream.write_label(label_while_exp)

        # (expression)
        self.xml_stream.write_tag_value(SYMBOL, LEFT_PAREN)
        self._eat(LEFT_PAREN)
        self.compile_expression()
        self.xml_stream.write_tag_value(SYMBOL, RIGHT_PAREN)
        self._eat(RIGHT_PAREN)

        self.vm_stream.write_arithmetic("not")
        self.vm_stream.write_if_goto(label_while_end)

        # {statements}
        self._handle_statements_within_braces()
        self.vm_stream.write_goto(label_while_exp)

        self.vm_stream.write_label(label_while_end)

        # <whileStatement>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('whileStatement')

    def compile_do_statement(self):
        """
        compile jack do statement
        """

        # <doStatement>
        self.xml_stream.write_open_tag('doStatement')
        self.xml_stream.reindent()

        # do
        self.xml_stream.write_tag_value(KEYWORD, DO)
        self._eat(DO)

        # TODO: could be refactored
        # subroutineName | (className | varName)'.'subroutineName
        subroutine_full_name = self.current_token_value
        # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
        self._eat(IDENTIFIER)
        # check if '.'
        if self.current_token_value == DOT:
            # self.xml_stream.write_tag_value(SYMBOL, DOT)
            self._eat(DOT)
            subroutine_full_name += f'{DOT}{self.current_token_value}'
            # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
            self._eat(IDENTIFIER)

        # now subroutine is (className | varName).subroutineName
        self._handle_subroutine_call_xml(subroutine_full_name)
        # (expressionList)
        self.xml_stream.write_tag_value(SYMBOL, LEFT_PAREN)
        self._eat(LEFT_PAREN)

        n_args = self.compile_expression_list()
        self.vm_stream.write_call(name=subroutine_full_name, n_args=n_args)
        # discard void functions return value (I guess only relevant in do statements)
        self.vm_stream.write_pop("temp", 0)

        self.xml_stream.write_tag_value(SYMBOL, RIGHT_PAREN)
        self._eat(RIGHT_PAREN)

        # ;
        self.xml_stream.write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

        # </doStatement>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('doStatement')

    def _handle_subroutine_call_xml(self, subroutine_full_name):
        obj, subroutine = subroutine_full_name.split('.') if '.' in subroutine_full_name else ['', subroutine_full_name]
        if self.symbol_table.contains(obj):
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.xml_stream.write_tag_value(tags={
                "name": obj,
                "kind": self.symbol_table.kind_of(obj),
                "index": self.symbol_table.index_of(obj),
                "context": "use"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)
        else:  # className
            self.xml_stream.write_open_tag(IDENTIFIER)
            self.xml_stream.reindent()
            self.xml_stream.write_tag_value(tags={
                "name": obj,
                "kind": CLASS,
                "context": "use"
            })
            self.xml_stream.deindent()
            self.xml_stream.write_close_tag(IDENTIFIER)
        self.xml_stream.write_tag_value(SYMBOL, DOT)
        self.xml_stream.write_open_tag(IDENTIFIER)
        self.xml_stream.reindent()
        self.xml_stream.write_tag_value(tags={
            "name": subroutine,
            "kind": SUBROUTINE,
            "context": "use"
        })
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag(IDENTIFIER)

    def compile_return_statement(self):
        """
        compile jack return statement
        """
        # <returnStatement>
        self.xml_stream.write_open_tag('returnStatement')
        self.xml_stream.reindent()

        # return
        self.xml_stream.write_tag_value(KEYWORD, RETURN)
        self._eat(RETURN)

        # expression?
        if self.current_token_value == SEMI_COLON:
            # every func in jack must return something (we use 0 as dummy return)
            self.vm_stream.write_push("constant", 0)
        else:
            self.compile_expression()

        self.vm_stream.write_return()

        # ;
        self.xml_stream.write_tag_value(SYMBOL, SEMI_COLON)
        self._eat(SEMI_COLON)

        # </returnStatement>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('returnStatement')

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
        self.xml_stream.write_open_tag('expression')
        self.xml_stream.reindent()

        # term
        self.compile_term()

        # (op term)*
        while self.current_token_value in OP:
            operation = VM_OPERATIONS.get(self.current_token_value, self.current_token_value)
            self.xml_stream.write_tag_value(SYMBOL, self.current_token_value)
            self._eat(self.current_token_value)
            self.compile_term()
            self.vm_stream.write_arithmetic(operation)

        # </expression>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('expression')

    def compile_term(self):
        """
        compile jack term
        """

        # <term>
        self.xml_stream.write_open_tag('term')
        self.xml_stream.reindent()

        current_token_value, current_token_type = self.current_token_value, self.current_token_type
        if current_token_type == INT_CONSTANT:
            self.vm_stream.write_push("constant", current_token_value)
            self.xml_stream.write_tag_value('integerConstant', current_token_value)
            self._eat(INT_CONSTANT)
        elif current_token_type == STR_CONSTANT:
            current_value = current_token_value
            self.vm_stream.write_push("constant", len(current_value))
            self.vm_stream.write_call("String.new", 1)
            for c in current_value:
                self.vm_stream.write_call("String.appendChar", 2)
                self.vm_stream.write_push("constant", ord(c))
            self.xml_stream.write_tag_value('stringConstant', current_token_value.strip(DOUBLE_QUOTES))
            self._eat(STR_CONSTANT)
        elif current_token_value in KEYWORD_CONSTANT:
            self.xml_stream.write_tag_value(KEYWORD, current_token_value)
            self._eat(current_token_value)
        elif current_token_value in UNARY_OP:
            self.xml_stream.write_tag_value(SYMBOL, current_token_value)
            self._eat(current_token_value)
            self.compile_term()
        elif current_token_value == LEFT_PAREN:  # '(' expression ')'
            self.xml_stream.write_tag_value(SYMBOL, LEFT_PAREN)
            self._eat(LEFT_PAREN)
            self.compile_expression()
            self.xml_stream.write_tag_value(SYMBOL, RIGHT_PAREN)
            self._eat(RIGHT_PAREN)
        else:  # identifier
            current_token_value = self.current_token_value
            self._eat(IDENTIFIER)
            next_token_value = self.current_token_value

            # subroutineCall: bar(expressionList)
            if next_token_value == LEFT_PAREN:
                self._handle_subroutine_call_xml(current_token_value)
                # self.xml_stream.write_tag_value(IDENTIFIER, current_token_value)
                self.xml_stream.write_tag_value(SYMBOL, LEFT_PAREN)
                self._eat(LEFT_PAREN)
                n_args = self.compile_expression_list()
                self.vm_stream.write_call(name=current_token_value, n_args=n_args)
                self.xml_stream.write_tag_value(SYMBOL, RIGHT_PAREN)
                self._eat(RIGHT_PAREN)

            # subroutineCall: foo.bar(expressionList) | Foo.bar(expressionList)
            elif next_token_value == DOT:
                if not (current_token_value == self.jack_tokenizer.src_base_name
                        or str.isupper(current_token_value[0])):
                    if self.symbol_table.contains(current_token_value):
                        self._handle_lookup_var_xml(current_token_value)
                        self.vm_stream.write_push(
                            self.symbol_table.kind_of(current_token_value),
                            self.symbol_table.index_of(current_token_value)
                        )
                    else:
                        raise CompileException(f"{current_token_value} is not defined")
                # TODO: handle this # (Foo|Baz).bar(expressionList)

                subroutine_full_name = current_token_value + DOT
                # self.xml_stream.write_tag_value(IDENTIFIER, current_token_value)
                # self.xml_stream.write_tag_value(SYMBOL, DOT)
                self._eat(DOT)
                # self.xml_stream.write_tag_value(IDENTIFIER, self.current_token_value)
                subroutine_full_name += self.current_token_value
                self._handle_subroutine_call_xml(subroutine_full_name)
                self._eat(IDENTIFIER)

                self.xml_stream.write_tag_value(SYMBOL, LEFT_PAREN)
                self._eat(LEFT_PAREN)
                n_args = self.compile_expression_list()
                self.vm_stream.write_call(name=subroutine_full_name, n_args=n_args)
                self.xml_stream.write_tag_value(SYMBOL, RIGHT_PAREN)
                self._eat(RIGHT_PAREN)

            # varName'[' expression ']'
            elif next_token_value == LEFT_BRACKET:
                if self.symbol_table.contains(current_token_value):
                    self._handle_lookup_var_xml(current_token_value)
                    self.vm_stream.write_push(
                        self.symbol_table.kind_of(current_token_value),
                        self.symbol_table.index_of(current_token_value)
                    )
                else:
                    raise CompileException(f"{current_token_value} is not defined")

                # self.xml_stream.write_tag_value(IDENTIFIER, current_token_value)
                self.xml_stream.write_tag_value(SYMBOL, LEFT_BRACKET)
                self._eat(LEFT_BRACKET)
                self.compile_expression()
                self.xml_stream.write_tag_value(SYMBOL, RIGHT_BRACKET)
                self._eat(RIGHT_BRACKET)
            # foo
            else:
                if self.symbol_table.contains(current_token_value):
                    self._handle_lookup_var_xml(current_token_value)
                    self.vm_stream.write_push(
                        self.symbol_table.kind_of(current_token_value),
                        self.symbol_table.index_of(current_token_value)
                    )
                else:
                    raise CompileException(f"{current_token_value} is not defined")
                # self.xml_stream.write_tag_value(IDENTIFIER, current_token_value)

        # </term>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('term')

    def _handle_lookup_var_xml(self, var):
        self.xml_stream.write_open_tag(IDENTIFIER)
        self.xml_stream.reindent()
        self.xml_stream.write_tag_value(tags={
            "name": var,
            "kind": self.symbol_table.kind_of(var),
            "index": self.symbol_table.index_of(var),
            "context": "use"
        })
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag(IDENTIFIER)

    def compile_expression_list(self):
        """
        compile jack expression list and return number of expressions compiled
        """
        # <expressionList>
        self.xml_stream.write_open_tag('expressionList')
        self.xml_stream.reindent()

        # (expression (',' expression)*)?
        n_expressions = 0
        if not self.current_token_value == RIGHT_PAREN:
            while True:
                self.compile_expression()
                n_expressions += 1
                if not self.current_token_value == COMMA:
                    break
                self.xml_stream.write_tag_value(SYMBOL, COMMA)
                self._eat(COMMA)

        # </expressionList>
        self.xml_stream.deindent()
        self.xml_stream.write_close_tag('expressionList')

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
        self._class_level_index = 0
        self._sub_level_data = []
        self._sub_level_index = 0

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
        self._sub_level_index = 0

    # defines a new identifier of given parameters and assign it a running index
    def define(self, name: str, _type: str, kind: Optional[str]):
        for arg in name, _type, kind:
            if not arg:
                raise CompileException(f"'{arg}' is not a valid variable name")

        if kind in (FIELD, STATIC):
            j_var = JVariable(name, _type, kind, self._class_level_index)
            self._class_level_data.append(j_var)
            self._class_level_index += 1
        if kind in (LOCAL, ARGUMENT):
            j_var = JVariable(name, _type, kind, self._sub_level_index)
            self._sub_level_index += 1
            self._sub_level_data.append(j_var)

        print(f"{name} added to symbol table")
        print(self.__str__())

    def var_count(self, kind: str):
        """
        return the number of variables of the given kind
        :param kind: kind of jack variable
        :return: number of variables of kind in symbol table
        """
        if kind in (FIELD, STATIC):
            return sum(kind is var.kind for var in self._class_level_data)
        if kind in (LOCAL, ARGUMENT):
            return sum(kind is var.kind for var in self._sub_level_data)
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


# TODO: adjust to output vm files (maybe keep xml too)
# Module 1: Jack compiler (ui)
def handle_file(src_path):
    logging.info(f'compiling {src_path}')
    out_xml_path = src_path.replace(SRC_FILE_EXT, XML_FILE_EXT)
    out_vm_path = src_path.replace(SRC_FILE_EXT, VM_FILE_EXT)
    with open(src_path) as src_stream, \
            open(out_xml_path, 'w') as xml_stream, \
            open(out_vm_path, 'w') as vm_stream:
        compilation_engine = CompilationEngine(JackTokenizer(src_stream), XMLWriter(xml_stream), VMWriter(vm_stream))
        compilation_engine.compile_class()
    logging.info(f'generated {out_xml_path}')
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
