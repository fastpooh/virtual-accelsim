/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_ELF_EUIJUN_CASL_ACCEL_SIM_ACCEL_SIM_FRAMEWORK_GPU_SIMULATOR_GPGPU_SIM_BUILD_GCC_11_4_0_CUDA_12040_RELEASE_CUOBJDUMP_TO_PTXPLUS_ELF_PARSER_HH_INCLUDED
# define YY_ELF_EUIJUN_CASL_ACCEL_SIM_ACCEL_SIM_FRAMEWORK_GPU_SIMULATOR_GPGPU_SIM_BUILD_GCC_11_4_0_CUDA_12040_RELEASE_CUOBJDUMP_TO_PTXPLUS_ELF_PARSER_HH_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int elf_debug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    C1BEGIN = 258,                 /* C1BEGIN  */
    C14BEGIN = 259,                /* C14BEGIN  */
    CMEMVAL = 260,                 /* CMEMVAL  */
    SPACE2 = 261,                  /* SPACE2  */
    C0BEGIN = 262,                 /* C0BEGIN  */
    STBEGIN = 263,                 /* STBEGIN  */
    STHEADER = 264,                /* STHEADER  */
    NUMBER = 265,                  /* NUMBER  */
    HEXNUMBER = 266,               /* HEXNUMBER  */
    IDENTIFIER = 267,              /* IDENTIFIER  */
    LOCALMEM = 268                 /* LOCALMEM  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 43 "elf.y"

	char* string_value;

#line 81 "/euijun-casl/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/build/gcc-11.4.0/cuda-12040/release/cuobjdump_to_ptxplus/elf_parser.hh"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE elf_lval;


int elf_parse (void);


#endif /* !YY_ELF_EUIJUN_CASL_ACCEL_SIM_ACCEL_SIM_FRAMEWORK_GPU_SIMULATOR_GPGPU_SIM_BUILD_GCC_11_4_0_CUDA_12040_RELEASE_CUOBJDUMP_TO_PTXPLUS_ELF_PARSER_HH_INCLUDED  */
