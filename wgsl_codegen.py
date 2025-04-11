import os
import textwrap
from io import StringIO

from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function
from wgsl_print import WGSLPrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, MatrixBase,
                            MatrixExpr)
from sympy.utilities.iterables import is_sequence
from sympy.utilities.codegen import CodeGenError,InOutArgument,Result,CodeGen,InputArgument,CodeGenArgumentListError,Routine,header_comment,ResultBase,OutputArgument
from sympy.printing.codeprinter import AssignmentError
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)



class WgslCodeGen(CodeGen):
    """Generator for WGSL code.

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.wgsl

    """

    code_extension = "wgsl"

    def __init__(self, project="project",documentation=None, cse=False):
        super().__init__(project=project, cse=cse)
        self.printer = WGSLPrinter()
        self.documentation = documentation

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append(" *%s*\n" % line.center(76))
        code_lines.append(" " + "*"*78 + "/\n")
        return code_lines


    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototypef

        """
        results = [self.get_datatype(i) for i in routine.results]

        if len(results) == 1:
            rstype = " -> " + results[0]
        elif len(routine.results) > 1:
            raise CodeGenError("Multiple return values not supported in Wgsl")
        else:
            rstype = ""

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if isinstance(arg, ResultBase):
                type_args.append((name, f"ptr<function,{self.get_datatype(arg)}>"))
            else:
                type_args.append((name, self.get_datatype(arg)))
        arguments = ", ".join([ "%s: %s" % t for t in type_args])
        return "fn %s(%s)%s" % (routine.name, arguments, rstype)
    
    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        documentation = ""
        if self.documentation:
            documentation = "/// %s\n" % self.documentation
        return ["%s%s {\n" % (documentation,prototype)]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):
        
        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        code_lines = []
        for result in routine.local_vars:

            # local variables that are simple symbols such as those used as indices into
            # for loops are defined declared elsewhere.
            if not isinstance(result, Result):
                continue

            if result.name != result.result_var:
                raise CodeGen("Result variable and name should match: {}".format(result))
            assign_to = result.name
            if isinstance(result.expr, (MatrixBase, MatrixExpr)):
                dims = result.expr.shape
                code_lines.append("let {}[{}];\n".format( str(assign_to), dims[0]*dims[1]))
                prefix = ""
            else:
                prefix = "let "

            constants, not_c, c_expr = self._printer_method_with_settings(
                'doprint', {"human": False, "dereference": dereference, "strict": False},
                result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))

            code_lines.append("{}{}\n".format(prefix, c_expr))

        return code_lines

    def _call_printer(self, routine):
        code_lines = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, (ResultBase,OutputArgument)):
                dereference.append(arg  .name)
                
                
        return_val = None
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name + "_result"
                t = self.get_datatype(result)
                code_lines.append("var {}:{};\n".format(str(assign_to),t))
                return_val = assign_to
            else:
                assign_to = result.result_var

            try:
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (self.get_datatype(result), str(assign_to)))
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', {"human": False, "dereference": dereference, "strict": False},
                    result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        return code_lines

    def _get_routine_ending(self, routine):
        return ["}\n"]

    def dump_c(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    dump_c.extension = code_extension  # type: ignore
    dump_c.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_c]
    
    
    def get_datatype(self, obj):
        """Returns the datatype of the given object.

        This is a simple wrapper around the printer's get_datatype method.
        """
        dtype = "f32"
        if isinstance(obj.name, MatrixSymbol):
            if obj.name.cols < 5 and obj.name.rows < 5:
                if obj.name.cols == 1:
                    return "vec%d<%s>" % (obj.name.rows, dtype)
                return "mat%dx%d<%s>" % (obj.name.rows, obj.name.cols,dtype)
            else:
                if obj.name.cols == 1:
                    return "array<%s,%d>" % (dtype,obj.name.rows)
                return "array<array<%s,%d>,%d>" % (dtype,obj.name.cols, obj.name.rows)
        return dtype


    def routine(self, name, expr, argument_sequence=None, global_vars=None):
        """Creates an Routine object that is appropriate for this language.

        This implementation is appropriate for at least C/Fortran.  Subclasses
        can override this if necessary.

        Here, we assume at most one return value (the l-value) which must be
        scalar.  Additional outputs are OutputArguments (e.g., pointers on
        right-hand-side or pass-by-reference).  Matrices are always returned
        via OutputArguments.  If ``argument_sequence`` is None, arguments will
        be ordered alphabetically, but with all InputArguments first, and then
        OutputArgument and InOutArguments.

        """

        if self.cse:
            from sympy.simplify.cse_main import cse

            if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
                if not expr:
                    raise ValueError("No expression given")
                for e in expr:
                    if not e.is_Equality:
                        raise CodeGenError("Lists of expressions must all be Equalities. {} is not.".format(e))

                # create a list of right hand sides and simplify them
                rhs = [e.rhs for e in expr]
                common, simplified = cse(rhs)

                # pack the simplified expressions back up with their left hand sides
                expr = [Equality(e.lhs, rhs) for e, rhs in zip(expr, simplified)]
            else:
                if isinstance(expr, Equality):
                    common, simplified = cse(expr.rhs) #, ignore=in_out_args)
                    expr = Equality(expr.lhs, simplified[0])
                else:
                    common, simplified = cse(expr)
                    expr = simplified

            local_vars = [Result(b,a) for a,b in common]
            local_symbols = {a for a,_ in common}
            local_expressions = Tuple(*[b for _,b in common])
        else:
            local_expressions = Tuple()


        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)
        
        if self.cse:
            if {i.label for i in expressions.atoms(Idx)} != set():
                raise CodeGenError("CSE and Indexed expressions do not play well together yet")
        else:
            # local variables for indexed expressions
            local_vars = {i.label for i in expressions.atoms(Idx)}
            local_symbols = local_vars

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        symbols = (expressions.free_symbols | local_expressions.free_symbols) - local_symbols - global_vars
        new_symbols = set()
        new_symbols.update(symbols)

        for symbol in symbols:
            if isinstance(symbol, Idx):
                new_symbols.remove(symbol)
                new_symbols.update(symbol.args[1].free_symbols)
            if isinstance(symbol, Indexed):
                new_symbols.remove(symbol)
        symbols = new_symbols

        # Decide whether to use output argument or return value
        return_val = []
        output_args = []
        for expr in expressions:
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                elif isinstance(out_arg, Symbol):
                    dims = []
                    symbol = out_arg
                elif isinstance(out_arg, MatrixSymbol):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg
                else:
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                if expr.has(symbol):
                    output_args.append(
                        InOutArgument(symbol, out_arg, expr, dimensions=dims))
                else:
                    output_args.append(
                        OutputArgument(symbol, out_arg, expr, dimensions=dims))

                # remove duplicate arguments when they are not local variables
                if symbol not in local_vars:
                    # avoid duplicate arguments
                    symbols.remove(symbol)
            # elif isinstance(expr, (ImmutableMatrix, MatrixSlice)):
            #     # Create a "dummy" MatrixSymbol to use as the Output arg
            #     out_arg = MatrixSymbol('out_%s' % abs(hash(expr)), *expr.shape)
            #     dims = tuple([(S.Zero, dim - 1) for dim in out_arg.shape])
            #     output_args.append(
            #         OutputArgument(out_arg, out_arg, expr, dimensions=dims))
            else:
                return_val.append(Result(expr))

        arg_list = []

        # setup input argument list

        # helper to get dimensions for data for array-like args
        def dimensions(s):
            return [(S.Zero, dim - 1) for dim in s.shape]

        array_symbols = {}
        for array in expressions.atoms(Indexed) | local_expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol) | local_expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            if symbol in array_symbols:
                array = array_symbols[symbol]
                metadata = {'dimensions': dimensions(array)}
            else:
                metadata = {}
            if symbol not in global_vars:
                arg_list.append(InputArgument(symbol, **metadata))

        output_args.sort(key=lambda x: str(x.name))
        arg_list.extend(output_args)

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    if isinstance(symbol, (IndexedBase, MatrixSymbol)):
                        metadata = {'dimensions': dimensions(symbol)}
                    else:
                        metadata = {}
                    new_args.append(InputArgument(symbol, **metadata))
            arg_list = new_args

        return Routine(name, arg_list, return_val, local_vars, global_vars)
