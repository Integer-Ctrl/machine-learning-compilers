Assembly
========

Before we begin implementing the individual components of the project, we will start with a brief review of assembly language.
This short chapter is intended as a refresher on the basic knowledge required for the project.

All files related to the tasks of this chapter can be found under ``submissions/assembly/``.

Hello Assembly
--------------

First, we will look at a simple assembly function, compile it, and examine the generated assembly code.

1. Compile
^^^^^^^^^^

**Task**: Use the GCC and Clang compilers to compile the function and generate assembly code.

The following commands will generate assembly code for the file ``hello_assembly.c``:

- gcc: ``gcc -S hello_assembly.c -o hello_assembly.s``
- clang: ``clang -S hello_assembly.c -o hello_assembly_clang.s``

2. Analyse
^^^^^^^^^^

**Task**: In the generated assembly code generated by the two compilers:

``gcc``

* **Subtask**: Locate the string "Hello Assembly Language!".

  - line 7

* **Subtask**: Identify the instructions that the compilers insert to conform to the procedure call standard.

  - ``stp x29, x30, [sp, -16]!``
  - ``mov x29, sp``
  - ``ldp x29, x30, [sp], 16``

* **Subtask**: Identify the function call to libc that prints the string.

  - ``bl puts`` (`branch link <https://developer.arm.com/documentation/dui0379/e/arm-and-thumb-instructions/bl>`_ and `puts <https://pubs.opengroup.org/onlinepubs/009695399/functions/puts.html>`_)

.. code-block:: asm
  :linenos:

    .arch armv8-a
    .file	"hello_assembly.c"
    .text
    .section	.rodata
    .align	3
  .LC0:
    .string	"Hello Assembly Language!"
    .text
    .align	2
    .global	hello_assembly
    .type	hello_assembly, %function
  hello_assembly:
  .LFB0:
    .cfi_startproc
    stp	x29, x30, [sp, -16]!
    .cfi_def_cfa_offset 16
    .cfi_offset 29, -16
    .cfi_offset 30, -8
    mov	x29, sp
    adrp	x0, .LC0
    add	x0, x0, :lo12:.LC0
    bl	puts
    nop
    ldp	x29, x30, [sp], 16
    .cfi_restore 30
    .cfi_restore 29
    .cfi_def_cfa_offset 0
    ret
    .cfi_endproc
  .LFE0:
    .size	hello_assembly, .-hello_assembly
    .ident	"GCC: (GNU) 14.2.1 20250110 (Red Hat 14.2.1-7)"
    .section	.note.GNU-stack,"",@progbits


``clang``

* **Subtask**: Locate the string "Hello Assembly Language!".

  - line 31

* **Subtask**: Identify the instructions that the compilers insert to conform to the procedure call standard.

  - ``stp x29, x30, [sp, #-16]!``
  - ``mov x29, sp``
  - ``ldp x29, x30, [sp], #16``

* **Subtask**: Identify the function call to libc that prints the string.

  - ``bl printf`` `branch link <https://developer.arm.com/documentation/dui0379/e/arm-and-thumb-instructions/bl>`_

.. code-block:: asm
  :linenos:

    .text
    .file	"hello_assembly.c"
    .globl	hello_assembly                  // -- Begin function hello_assembly
    .p2align	2
    .type	hello_assembly,@function
  hello_assembly:                         // @hello_assembly
    .cfi_startproc
  // %bb.0:
    stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
    .cfi_def_cfa_offset 16
    mov	x29, sp
    .cfi_def_cfa w29, 16
    .cfi_offset w30, -8
    .cfi_offset w29, -16
    adrp	x0, .L.str
    add	x0, x0, :lo12:.L.str
    bl	printf
    .cfi_def_cfa wsp, 16
    ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
    .cfi_def_cfa_offset 0
    .cfi_restore w30
    .cfi_restore w29
    ret
  .Lfunc_end0:
    .size	hello_assembly, .Lfunc_end0-hello_assembly
    .cfi_endproc
                                          // -- End function
    .type	.L.str,@object                  // @.str
    .section	.rodata.str1.1,"aMS",@progbits,1
  .L.str:
    .asciz	"Hello Assembly Language!\n"
    .size	.L.str, 26

    .ident	"clang version 19.1.7 (Fedora 19.1.7-3.fc41)"
    .section	".note.GNU-stack","",@progbits
    .addrsig
    .addrsig_sym printf


3. Driver
^^^^^^^^^

**Task**: Write a C++ driver that calls the ``hello_assembly`` function and test your implementation.

The driver code can be found in the file ``hello_assembly.cpp``:

.. code-block:: cpp

  extern "C"
  {
    void hello_assembly();
  }

  int main()
  {
    hello_assembly();
    return 0;
  }

Commands to generate an executable and run it:

- ``gcc -c hello_assembly.c -o hello_assembly.o``
- ``g++ -o hello_assembly.exe hello_assembly.cpp hello_assembly.o``
- .. image:: ../_static/images/report_25_04_17/hello_assembly_example.png
    :align: center


Assembly Function
-----------------

Next we have a look at the assembly file ``add_values.s`` which contains a function that adds two values together.

1. Assemble
^^^^^^^^^^^

**Task**: Assemble the file and use the name ``add_values.o`` for the output.

- ``as add_values.s -o add_values.o``

2. Generate
^^^^^^^^^^^

**Task**: Generate the following from ``add_values.o``:

* **Subtask**: Hexadecimal dump

  - ``od -A x -t x1 add_values.o > add_values.hex``

* **Subtask**: Section Headers

  - ``readelf -S add_values.o > add_values.relf``

* **Subtask**: Disassembly

  - ``objdump --syms -S -d add_values.o > add_values.dis``

3. Analyse
^^^^^^^^^^

**Task**: Find the size of the ``.text`` section in the generated output and explain it.

.. code-block::
  :linenos:

  There are 7 section headers, starting at offset 0x130:

  Section Headers:
    [Nr] Name              Type             Address           Offset
        Size              EntSize          Flags  Link  Info  Align
    [ 0]                   NULL             0000000000000000  00000000
        0000000000000000  0000000000000000           0     0     0
    [ 1] .text             PROGBITS         0000000000000000  00000040
        0000000000000020  0000000000000000  AX       0     0     4
    [ 2] .data             PROGBITS         0000000000000000  00000060
        0000000000000000  0000000000000000  WA       0     0     1
    [ 3] .bss              NOBITS           0000000000000000  00000060
        0000000000000000  0000000000000000  WA       0     0     1
    [ 4] .symtab           SYMTAB           0000000000000000  00000060
        0000000000000090  0000000000000018           5     5     8
    [ 5] .strtab           STRTAB           0000000000000000  000000f0
        000000000000000f  0000000000000000           0     0     1
    [ 6] .shstrtab         STRTAB           0000000000000000  000000ff
        000000000000002c  0000000000000000           0     0     1
  Key to Flags:
    W (write), A (allocate), X (execute), M (merge), S (strings), I (info),
    L (link order), O (extra OS processing required), G (group), T (TLS),
    C (compressed), x (unknown), o (OS specific), E (exclude),
    D (mbind), p (processor specific)


Size of ``.text``: 0x20 byte or equal 32 bytes. ``.text`` corresponds to the size of all instructions. The add_values.s file has 8 instructions in total, each is 4 byte long. Therefore, :math:`8 \cdot4` byte :math:`=32` byte :math:`=` 0x20 byte.

4. Driver
^^^^^^^^^

**Task**: Write a C++ driver that calls the ``add_values`` function and illustrate it with an example.

The driver code can be found in the file ``add_values.cpp``:

- ``g++ -o add_values.exe add_values.cpp add_values.o``
- .. image:: ../_static/images/report_25_04_17/add_values_example.png
    :align: center


5. GDB
^^^^^^

**Task**: Use the GNU Project Debugger `GDB <https://www.sourceware.org/gdb/>`__ to step through an example call to the ``add_values`` function.
Display the contents of the general-purpose registers after each of the executed instructions.

Using GDB

  - ``gdb <executable>``
  - Inside GDB
  - ``lay next``

    - press \<Enter\> to toggle the available views of GDB
    - .. note::
        The current layer view will be fixed if an instruction is run.
        Use ``lay next`` to be able to toggle the views again.

  - Use following commands to navigate:

    - ``break <label>`` adds a breakpoint at a specific label e.g. a function declaration
    - ``run`` starts the program
    - ``next`` move to the next line in the C++ code
    - ``nexti`` move to the next line in the assembly Instruction
    - ``step`` step into a function call
    - ``ref`` refreshes the view
    - ``x/i $pc`` examines the Instruction at the program counter
    - ``info registers`` show the current register state of the program
    - ``quit`` exit GDB

- ``g++ -o add_values.exe -g add_values.cpp add_values.o`` Add debug information with ``-g`` 
- ``gdb add_values.exe``
- ``lay next``
- Press \<Enter\> 3 times to get a view with assembly instruction and the registers.
- ``break add_values``
- ``run``
- .. image:: ../_static/images/report_25_04_17/gdb_instruction01.png
    :align: center
- ``nexti``
- .. image:: ../_static/images/report_25_04_17/gdb_instruction02.png
    :align: center
- ``nexti``
- .. image:: ../_static/images/report_25_04_17/gdb_instruction03.png
    :align: center
- ``nexti``
- .. image:: ../_static/images/report_25_04_17/gdb_instruction04.png
    :align: center
- ``nexti``
- .. image:: ../_static/images/report_25_04_17/gdb_instruction05.png
    :align: center
- ``nexti``
- .. image:: ../_static/images/report_25_04_17/gdb_instruction06.png
    :align: center
