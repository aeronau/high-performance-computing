/*

High performance computing:

SOLUTION TO A HEAT TRANSFER SITUATION

M. E.
S. E.
Arnau Prat Gasull

Terrassa
June 2017

*/

/* NOTE - We wish we had not overestimated our possibilities (especially Arnau), but here is a program we are proud of in terms of parallel/multiproc computing (it does not compute the values correctly :), but that is not the important bit). We have thought of complex problems and developed complex algorithms (in terms of workload division accross procs) to tackle the problem only to finally use the same idea as A. M. and C. G. (not the same code, obviously), but instead of using multidimensional arrays, using unidimensional arrays, as taught in class. We have not had time to implement other ideas that would make the program beautiful and truly usable, but we are extremely satisfied of what we have learnt. */

/* NOTE - The MPI part has been thoroughtly tested with 4 procs and below you will find the remains of a possible future implementation for more procs (as all values have been computed using general variables and formulas, not numbers). We have run the program with more procs and sometimes it has run. Our main goal was to develop the halo update for 4 procs (the prints at the end are for the first 4 procs), which we have accomplished. */

/* NOTE - The program may run for NProcCOL != 2, but it has not been tested very much. In fact, after a few changes to improve support for NProcCOL != 2, the program has started to output different values (even for the same configuration as before). We do not have time to check why it happened, but we can provide older versions that output a better result. */

/* NOTE - The program will be updated ASAP so it can solve transient problems. */

/* NOTE - The code will be tidied up (multiple files) in the future, but we were required to submit only one file. */

/* TODO: In order to solve a transient situation, one can use the following: 

    AW(i, j) = k * VERT_SIDE / (X(i,j) - X(i - 1, j));
    AE(i, j) = k * VERT_SIDE / 2 / (X(i,j) - X(i + 1, j));
    AS(i, j) = k * HORI_SIDE / (X(i,j) - X(i, j - 1));
    AN(i, j) = k * HORI_SIDE / (X(i,j) - X(i, j + 1));
    AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) - dens * HORI_SIDE * VERT_SIDE * cp / Deltat;
    BP(i, j) = - dens * HORI_SIDE * VERT_SIDE * cp * T(i, j) / Deltat;

    (Result checked by Marc Espinos)

    It should be easy to implement, one new `for` may be required and another matrix to store the temperature map at each istant.

*/

/* IMPORTANT - Using SI values */

/* IMPORTANT - The halo is constituted by the nodes adjacent to the area to be solved minus the cornering ones */

/* INCLUDERES */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

/* MACROS */

/* Map 2D memory to 1D: access memory block using two indices */

/* Automatic allocation of memory at LOCAL indices achieved with the translator idxGLOBAL2LOCAL(...) */

#define access(address0, I, J) *(address0 + idxGLOBAL2LOCAL(I, procROW, M, NProcROW)*(Nmem + 2) + idxGLOBAL2LOCAL(J, procCOL, N, NProcCOL)) // Nmem + 2: number of points per side (Nmem) plus 1 halo point at each side (2)

/* Shortcut for material map */

#define element(I, J) access(numMap, I, J)

/* Shortcut for temperature map */

#define T(I, J) access(tempMap, I, J)
#define TO(I, J) access(oldTempMap, I, J)

/* Shortcut for coefficient matrices */

#define AS(I, J) access(as, I, J)
#define AN(I, J) access(an, I, J)
#define AW(I, J) access(aw, I, J)
#define AE(I, J) access(ae, I, J)
#define AP(I, J) access(ap, I, J)
#define BP(I, J) access(bp, I, J)

/* Shortcut for coordinate map */

#define X(I, J) access(xCollection, I, J)
#define Y(I, J) access(yCollection, I, J)

/* Simple functions */

#define MAXOF(a, b) (((a) >= (b)) ? (a) : (b))
#define MINOF(a, b) (((a) < (b)) ? (a) : (b))
#define ABSVALOF(a) ((a) > 0) ? (a) : (-(a)) 
#define SPEAK_PROC(a, b) if(proc == (b)) {printf("(%i):\t", (b)); (a); printf("\r\n");} // Proc b speaking
#define SPEAK(a) SPEAK_PROC(a, mainProc) // Delegate speaking

/* Shortcut for functions - Simplify function calls */

/* TODO - Not all parameters may be needed, check which ones are truly needed */

#define HALOUPDATE(A, t) haloUpdate(A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem, t)
#define SOLVERJACOBI(tempMap, oldTempMap, ae, aw, an, as, ap, bp) solverJacobi(tempMap, oldTempMap, ae, aw, an, as, ap, bp, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem, ite);

/* Shortcut for MPI send and recieve functions - HORIZONTAL TRANSMISSION */

/* TODO - Not all parameters may be needed, check which ones are truly needed */

#define SEND2RIGHT(A, t)    send2right(A, t, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)
#define SEND2LEFT(A, t)      send2left(A, t, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)
#define RECV0RIGHT(A, t, s) recv0right(A, t, s, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)
#define RECV0LEFT(A, t, s)   recv0left(A, t, s, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)

/* Shortcut for MPI send and recieve functions - VERTICAL TRANSMISSION */

/* TODO - Not all parameters may be needed, check which ones are truly needed */

#define SEND2TOP(A, t)          send2top(A, t, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)
#define SEND2BOTTOM(A, t)    send2bottom(A, t, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)
#define RECV0TOP(A, t, s)       recv0top(A, t, s, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)
#define RECV0BOTTOM(A, t, s) recv0bottom(A, t, s, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem)

/* FUNCTIONS */

/* TODO - Prototypes, obviously */

/* FUNCTION - procFirstValGLOBAL: returns the first GLOBAL index used by the proc */

int procFirstValGLOBAL(int proc, int N, int NProc) {

    int firstVal = proc * (N/NProc);

    return(firstVal);

}

/* FUNCTION - procLastValGLOBAL: returns the last GLOBAL index used by proc */

int procLastValGLOBAL(int proc, int N, int NProc) {

    int lastVal = procFirstValGLOBAL(proc + 1, N, NProc) - 1; // Last processor memory position is the memory position just before the first memory position for the next processor

    return(lastVal);

}

/* FUNCTION - memPosGLOBAL2LOCAL: returns the local index according to the global index */

int idxGLOBAL2LOCAL(int idxGLOBAL, int proc, int N, int NProc) {

    int idxLOCAL = idxGLOBAL - procFirstValGLOBAL(proc, N, NProc) + 1; // Used by access(address0, I, J)

    return(idxLOCAL);

}

/* FUNCTION - buildVerticalArray: returns all elements of column i so they can be sent */

double *buildVerticalArray(double *x, int M,  int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int i, int jSTART, int jEND, int Nmem) {

    double *verticalValsSend;
    verticalValsSend = malloc(sizeof(double)*(1 + jEND - jSTART));

    int j = jSTART;

    for(j; j < jEND + 1; j++) {

        verticalValsSend[j - jSTART] = access(x, i, j);

    }

    return verticalValsSend;

}

/* FUNCTION - storeVerticalArray: gets the values of the recieved column and allocates them in the matrix at column i */

void storeVerticalArray(double *verticalValsRecv, double *x, int M,  int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int i, int jSTART, int jEND, int Nmem) {
 
    int j = jSTART;

    for(j; j < jEND + 1; j++) {

        access(x, i, j) = verticalValsRecv[j - jSTART];

    }

}

/* FUNCTION - buildHorizontalArray: analog to buildVerticalArray */

double *buildHorizontalArray(double *x, int M,  int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int j, int Nmem) {

    double *horizontalValsSend;
    horizontalValsSend = malloc(sizeof(double)*(1 + iEND - iSTART));

    int i = iSTART;

    for(i; i < iEND + 1; i++) {

        horizontalValsSend[i - iSTART] = access(x, i, j);
    }

    return horizontalValsSend;

}

/* FUNCTION - storeHorizontalArray: analog to storeVerticalArray */

void storeHorizontalArray(double *horizontalValsRecv, double *x, int M,  int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int j, int Nmem) {
 
    int i = iSTART;

    for(i; i < iEND + 1; i++) {

        access(x, i, j) = horizontalValsRecv[i - iSTART];

    }

}

/* FUNCTION - send2top: gets the top row of the matrix and sends it to the halo of the proc "on top" */

void send2top(double *A, int t, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *horizontalValsSend;
    horizontalValsSend = buildHorizontalArray(A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jEND, Nmem);

    MPI_Ssend(horizontalValsSend, 1 + iEND - iSTART, MPI_DOUBLE, proc + NProcROW, t, MPI_COMM_WORLD);

    free(horizontalValsSend);

}

/* FUNCTION - send2bottom: analog to send2top */

void send2bottom(double *A, int t, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *horizontalValsSend;
    horizontalValsSend = buildHorizontalArray(A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, Nmem);

    MPI_Ssend(horizontalValsSend, 1 + iEND - iSTART, MPI_DOUBLE, proc - NProcROW, t, MPI_COMM_WORLD);

    free(horizontalValsSend);

}

/* FUNCTION - recv0top: recieves the row sent by the "bottom" proc and stores it in the corresponding halo */

void recv0top(double *A, int t, MPI_Status s, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *horizontalValsRecv;
    horizontalValsRecv = malloc(sizeof(double)*(1 + iEND - iSTART));

    MPI_Recv(horizontalValsRecv, 1 + iEND - iSTART, MPI_DOUBLE, proc + NProcROW, t, MPI_COMM_WORLD, &s);

    storeHorizontalArray(horizontalValsRecv, A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jEND + 1, Nmem);

    free(horizontalValsRecv);

}

/* FUNCTION - recv0bottom: analog to recv0top */

void recv0bottom(double *A, int t, MPI_Status s, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *horizontalValsRecv;
    horizontalValsRecv = malloc(sizeof(double)*(1 + iEND - iSTART));

    MPI_Recv(horizontalValsRecv, 1 + iEND - iSTART, MPI_DOUBLE, proc - NProcROW, t, MPI_COMM_WORLD, &s);

    storeHorizontalArray(horizontalValsRecv, A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART - 1, Nmem);

    free(horizontalValsRecv);

}

/* FUNCTION - send2right: analog to send2top */

void send2right(double *A, int t, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *verticalValsSend;
    verticalValsSend = buildVerticalArray(A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iEND, jSTART, jEND, Nmem);

    MPI_Ssend(verticalValsSend, 1 + jEND - jSTART, MPI_DOUBLE, proc + 1, t, MPI_COMM_WORLD);

    free(verticalValsSend);

}

/* FUNCTION - send2left: analog to send2top */

void send2left(double *A, int t, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *verticalValsSend;
    verticalValsSend = buildVerticalArray(A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, jSTART, jEND, Nmem);

    MPI_Ssend(verticalValsSend, 1 + jEND - jSTART, MPI_DOUBLE, proc - 1, t, MPI_COMM_WORLD);

    free(verticalValsSend);

}

/* FUNCTION - recv0right: analog to recv0top */

void recv0right(double *A, int t, MPI_Status s, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *verticalValsRecv;
    verticalValsRecv = malloc(sizeof(double)*(1 + jEND - jSTART));

    MPI_Recv(verticalValsRecv, 1 + jEND - jSTART, MPI_DOUBLE, proc + 1, t, MPI_COMM_WORLD, &s);

    storeVerticalArray(verticalValsRecv, A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iEND + 1, jSTART, jEND, Nmem);

    free(verticalValsRecv);

}

/* FUNCTION - recv0left: analog to recv0top */

void recv0left(double *A, int t, MPI_Status s, int M, int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    double *verticalValsRecv;
    verticalValsRecv = malloc(sizeof(double)*(1 + jEND - jSTART));

    MPI_Recv(verticalValsRecv, 1 + jEND - jSTART, MPI_DOUBLE, proc - 1, t, MPI_COMM_WORLD, &s);

    storeVerticalArray(verticalValsRecv, A, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART - 1, jSTART, jEND, Nmem);

    free(verticalValsRecv);

}

/* FUNCTION - printer: correctly prints a matrix of interest */

void printer(double *a, int M,  int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem) {

    int i;
    int j = 0;

    for(j; j < 3 + jEND - jSTART; j++) {

        i = iSTART - 1;

        for(i; i < 2 + iEND; i++) {

            printf("%08.2f\t", access(a, i, 1 + jEND - j)); // j + jEND - j allows the program to print the result coherently with the axes

        }

        printf("(%i)\r\n\t", proc);

    }

}

/* FUNCTION - checkr: stolen function */

void checkr(int r,char *txt) {

    if(r != MPI_SUCCESS) {

        fprintf(stderr, "Error: %s\n", txt);
        exit(-1);

    }
}

/* FUNCTION - procAmI: stolen function with name changed */

int procAmI() {

    int me;
    checkr(MPI_Comm_rank(MPI_COMM_WORLD, &me), "procAmI()");
    return(me);

}

/* FUNCTION - numProc: stolen function with name changed */

int numProc() {

    int total;
    checkr(MPI_Comm_size(MPI_COMM_WORLD, &total), "numProc()");
    return(total);

}

/* FUNCTION - matrixMemAlloc and check: stolen function with name changed */

double *matrixMemAlloc(int N) {

    double *r;
    r = malloc(N);

    if(r == NULL) {

        printf("mem\n");
        MPI_Finalize();
        exit(-1);

    }

    return(r);
}

/* FUNCTION - solverJacobi: solves the matrix using the coefficient matrices */

void solverJacobi(double *tempMap, double *oldTempMap, double *ae, double *aw, double *an, double *as, double *ap, double *bp, int M,  int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem, int ite) {

    int j = jSTART;

    for(j; j < jEND + 1; j++) {

        int i = iSTART;

        for(i; i < iEND + 1; i++) {

            // printf("(%i): %f K ", proc, T(i, j));

            T(i, j) = (AE(i, j) * TO(i + 1, j) + AW(i, j) * TO(i - 1, j) + AS(i, j) * TO(i, j - 1) + AN(i, j) * TO(i, j + 1) + BP(i, j)) / AP(i, j);

            // printf(" -> %f K (ite: %i)\n", T(i, j), ite);

        }
    }
}

/* FUNCTION - copyArr: copies an array */

void copyArr(double *oldTempMap, double *tempMap, int Nmem) {

    int i = 0;

    for(i; i < Nmem*Nmem + 1; i++) {

        oldTempMap[i] = tempMap[i];

    }
}

/* FUNCTION - haloUpdate: updates the values of the halo with the values of the procs */

int haloUpdate(double *x, int M,  int N, int proc, int procROW, int procCOL, int NProcROW, int NProcCOL, int iSTART, int iEND, int jSTART, int jEND, int Nmem, int t) {

    /* INIT - MPI info: status checkers... */

    int r;
    MPI_Status st;

    /* COMPUTE - Send and recieve data to and from processors according to a checkered domain (e.g. chessboard) */

    int color = (proc + (NProcROW%2 == 0 ? 1 : 0)*procCOL) % 2; // Color of the processor: 0 or 1 (e.g. white or black)

    if(color == 0) { // Proc with color 0 is first sender and second reciever

        /* BOTTOM LEFT */

        if((procROW == 0) && (procCOL == 0)) {

            /* Send to right proc */
            SEND2RIGHT(x, t);
            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);

            /* Send to top proc */
            SEND2TOP(x, t);
            /* Recieve from top proc */
            RECV0TOP(x, t, st);

        }

        /* BOTTOM MIDDLE */

        if((procROW != 0) && (procROW != NProcROW - 1) && (procCOL == 0)) {

            /* Send to left proc */
            SEND2LEFT(x, t);
            /* Recieve from  proc */
            RECV0LEFT(x, t, st);

            /* Send to right proc */
            SEND2RIGHT(x, t);
            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);

            /* Send to top proc */
            SEND2TOP(x, t);
            /* Recieve from top proc */
            RECV0TOP(x, t, st);

        }

        /* BOTTOM RIGHT */

        if((procROW == NProcROW - 1) && (procCOL == 0)) {

            /* Send to left proc */
            SEND2LEFT(x, t);
            /* Recieve from left proc */
            RECV0LEFT(x, t, st);

            /* Send to top proc */
            SEND2TOP(x, t);
            /* Recieve from top proc */
            RECV0TOP(x, t, st);

        }

        /* MIDDLE RIGHT */

        if((procROW == NProcROW - 1) && (procCOL != 0) && (procCOL != NProcCOL - 1)) {

            /* Send to left proc */
            SEND2LEFT(x, t);
            /* Recieve from left proc */
            RECV0LEFT(x, t, st);

            /* Send to bottom proc */
            SEND2BOTTOM(x, t);
            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);

            /* Send to top proc */
            SEND2TOP(x, t);
            /* Recieve from top proc */
            RECV0TOP(x, t, st);

        }

        /* TOP RIGHT */

        if((procROW == NProcROW - 1) && (procCOL == NProcCOL - 1)) {

            /* Send to left proc */
            SEND2LEFT(x, t);
            /* Recieve from left proc */
            RECV0LEFT(x, t, st);

            /* Send to bottom proc */
            SEND2BOTTOM(x, t);
            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);

        }

        /* TOP MIDDLE */

        if((procROW != 0) && (procROW != NProcROW - 1) && (procCOL == NProcCOL - 1)) {

            /* Send to left proc */
            SEND2LEFT(x, t);
            /* Recieve from  proc */
            RECV0LEFT(x, t, st);

            /* Send to right proc */
            SEND2RIGHT(x, t);
            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);

            /* Send to bottom proc */
            SEND2BOTTOM(x, t);
            /* Recieve from BOTTOM proc */
            RECV0BOTTOM(x, t, st);

        }

        /* TOP LEFT */

        if((procROW == 0) && (procCOL == NProcCOL - 1)) {

            /* Send to right proc */
            SEND2RIGHT(x, t);
            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);

            /* Send to bottom proc */
            SEND2BOTTOM(x, t);
            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);

        }

        /* MIDDLE LEFT */

        if((procROW == 0) && (procCOL != 0) && (procCOL != NProcCOL - 1)) {

            /* Send to bottom proc */
            SEND2BOTTOM(x, t);
            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);

            /* Send to right proc */
            SEND2RIGHT(x, t);
            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);

            /* Send to top proc */
            SEND2TOP(x, t);
            /* Recieve from top proc */
            RECV0TOP(x, t, st);

        }

    } else { // color = 1

        /* Bottom left cannot have a color of 1 */

        /* BOTTOM MIDDLE */

        if((procROW != 0) && (procROW != NProcROW - 1) && (procCOL == 0)) {

            /* Recieve from  proc */
            RECV0LEFT(x, t, st);
            /* Send to left proc */
            SEND2LEFT(x, t);

            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);
            /* Send to right proc */
            SEND2RIGHT(x, t);

            /* Recieve from top proc */
            RECV0TOP(x, t, st);
            /* Send to top proc */
            SEND2TOP(x, t);

        }

        /* BOTTOM RIGHT */

        if((procROW == NProcROW - 1) && (procCOL == 0)) {

            /* Recieve from left proc */
            RECV0LEFT(x, t, st);
            /* Send to left proc */
            SEND2LEFT(x, t);

            /* Recieve from top proc */
            RECV0TOP(x, t, st);
            /* Send to top proc */
            SEND2TOP(x, t);

        }

        /* MIDDLE RIGHT */

        if((procROW == NProcROW - 1) && (procCOL != 0) && (procCOL != NProcCOL - 1)) {

            /* Recieve from left proc */
            RECV0LEFT(x, t, st);
            /* Send to left proc */
            SEND2LEFT(x, t);

            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);
            /* Send to bottom proc */
            SEND2BOTTOM(x, t);

            /* Recieve from top proc */
            RECV0TOP(x, t, st);
            /* Send to top proc */
            SEND2TOP(x, t);

        }

        /* TOP RIGHT */

        if((procROW == NProcROW - 1) && (procCOL == NProcCOL - 1)) {

            /* Recieve from left proc */
            RECV0LEFT(x, t, st);
            /* Send to left proc */
            SEND2LEFT(x, t);

            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);
            /* Send to bottom proc */
            SEND2BOTTOM(x, t);

        }

        /* TOP MIDDLE */

        if((procROW != 0) && (procROW != NProcROW - 1) && (procCOL == NProcCOL - 1)) {

            /* Recieve from  proc */
            RECV0LEFT(x, t, st);
            /* Send to left proc*/
            SEND2LEFT(x, t);

            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);
            /* Send to right proc */
            SEND2RIGHT(x, t);

            /* Recieve from BOTTOM proc */
            RECV0BOTTOM(x, t, st);
            /* Send to bottom proc */
            SEND2BOTTOM(x, t);

        }

        /* TOP LEFT */

        if((procROW == 0) && (procCOL == NProcCOL - 1)) {

            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);
            /* Send to right proc*/
            SEND2RIGHT(x, t);

            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);
            /* Send to bottom proc */
            SEND2BOTTOM(x, t);

        }

        /* MIDDLE LEFT */

        if((procROW == 0) && (procCOL != 0) && (procCOL != NProcCOL - 1)) {

            /* Recieve from bottom proc */
            RECV0BOTTOM(x, t, st);
            /* Send to bottom proc */
            SEND2BOTTOM(x, t);

            /* Recieve from right proc */
            RECV0RIGHT(x, t, st);
            /* Send to right proc */
            SEND2RIGHT(x, t);

            /* Recieve from top proc */
            RECV0TOP(x, t, st);
            /* Send to top proc */
            SEND2TOP(x, t);

        }

    }

}

/* NOTE - In the following functions there used to be a switch which allowed the function to return different densities for different materials */

/* FUNCTION - density */

double density(int id) {
    
    /* DATA INPUT - Densities */

    return 1500.; // USER INPUT
}

/* FUNCTION - convectionHeatTransferCoef */

double convectionHeatTransferCoef(int id) {

    /* DATA INPUT - Convection heat transfer coefficient */

    return 9.; // USER INPUT

}

/* FUNCTION - conductionHeatTransferCoef */

double conductionHeatTransferCoef(int id) {

    /* DATA INPUT - Conduction heat transfer coefficient */

    return 170.; // USER INPUT

}

/* FUNCTION - specificHeatCoef */

double specificHeatCoef(int id) {

    /* DATA INPUT - Specific heat coefficient */
    
    return 750.; // USER INPUT

}

/* FUNCTION - main */

int main(int argc, char **argv) {

    /* INIT - MPI */

    checkr(MPI_Init(&argc,&argv),"init");

    /* DATA INPUT - Thermodinamic data */

    /* DATA INPUT - External Boundaries */

    double TLEFT = 300. + 273.; // USER INPUT
    double TBOTTOM = 350. + 273.; // USER INPUT
    double TRIGHT = 320. + 273.; // USER INPUT
    double TTOP = 380. + 273.; // USER INPUT

    /* DATA INPUT - Number of points */

    /* IMPORTANT - Points have to be divisible by number of processors per side */

    /* TODO - One could pass the arguments to check if the number of points is correct */

    int M = 8; // USER INPUT - Points x direction
    int N = 12; // USER INPUT - Points y direction

    /* DATA INPUT - Geometry: points of interest */

    double pX = 1.1, pY = 0.8; // USER INPUT - Top right point

    double dX = pX / (M - 1);
    double dY = pY / (N - 1);

    /* DATA INPUT - Numerical data */

    // double Deltat = 0.1; // USER INPUT for a transient situation
    // double tLIM = 100; // USER INPUT for a transient situation
    int iteMax = 100000; // USER INPUT
    double delta = 0.001; // USER INPUT

    /* VAR INIT - Numerical data */

    double diff, diffMax;
    int converged = 0; // Will be set to 1 if solution coverges (at error delta)
    int convergedSum; // Program will stop after all procs have converged (sum is 4)

    /* VAR INIT - Init counters */

    int i, j; // Point of domain selectors
    int ite = 1;

    /* VAR INIT - Element selector init */

    int id; // Material selector

    /* VAR INIT - Cell geometry variables */

    double HORI_SIDE; // Length of bottom and top sides (cell is rectangular)
    double VERT_SIDE; // Length of left and right sides (cell is rectangular)

    /* VAR INIT - Parallelization variables */

    int NProc = numProc(); // Number of procs has to be a product of two numbers
    int proc = procAmI(); // Proc rank
    int NProcCOL = 2; // USER INPUT - User fixes procs per col (may not work for NProcCOL != 2)
    int NProcROW = NProc / NProcCOL; // Number of procs per row
    int procROW = proc % NProcROW; // Proc rank in row
    int procCOL = (proc - procROW) * NProcCOL / NProc; // Proc rank in col
    int mainProc = 0; // USER INPUT - User fixes the delegate

    // SPEAK(printf((NProc != 1) ? "There are %i procs\r\n" : "There is %i proc\r\n" , NProc)); // Print info

    /* MEMORY ALLOCATION - Allocate memory block which will be accessed using two indices (see MACROS) */

    int Nmem = MAXOF(M / NProcROW, N / NProcCOL) + 2; // Number of memory positions per side (square, not rectangle) (CURRENT IMPLEMENTATION)

    double *numMap; // id map
    double *tempMap; // Temperature map
    double *oldTempMap; // Old temperature map
    double *as; // South coefficient map
    double *an; // North coefficient map
    double *aw; // West coefficient map
    double *ae; // East coefficient map
    double *ap; // Point coefficient (depedent variable) map
    double *bp; // Point coefficient (independent variable) map
    double *xCollection; // Point x coordinate map
    double *yCollection; // Point y coordinate map

    /* Memory management is the same for all matrices - All procs have the same halo, which will be feeded according to the procs' position */

    numMap = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    tempMap = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    oldTempMap = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    as = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    an = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    aw = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    ae = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    ap = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    bp = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    xCollection = matrixMemAlloc(sizeof(double)*Nmem*Nmem);
    yCollection = matrixMemAlloc(sizeof(double)*Nmem*Nmem);

    /* INIT - Thermodynamic variables */

    double k; // Conduction param
    double alpha; // Convection param
    // double dens; // Density - Needed for a transient problem
    // double cp; // CP - Needed for a transient problem

    /* INIT - Iterators */

    int iSTART;
    int iEND;
    int jSTART;
    int jEND;
    int compareIdx;

    /* COMPUTE - Divide region */

    /* Assingn values of iterators: proc now knows which GLOBAL idxs will use (and ultimately its position) */

    iSTART = procFirstValGLOBAL(procROW, M, NProc / NProcCOL); // Init i var iterated by proc
    iEND = procLastValGLOBAL(procROW, M, NProc / NProcCOL ); // Last i var iterated by proc

    jSTART = procFirstValGLOBAL(procCOL, N, NProcCOL); // Init j var iterated by proc
    jEND = procLastValGLOBAL(procCOL, N, NProcCOL); // Last j var iterated by proc

    // SPEAK_PROC(printf("Proc %i (%i in row, %i in col) idxs i = [%i ... %i] and j = [%i ... %i]\r\n", proc, procROW, procCOL, iSTART, iEND, jSTART, jEND), proc); // Print info

    /* TODO - The following was to be placed in a function outside the main, but because of the complexity of the problem and debug purposes, we have not had time to create the function. It should be easy, though */

    /* COMPUTE - Fill numMap matrix */

    i = iSTART;

    for(i; i < iEND + 1; i++) { // TODO - Reverse for(i..) for(j...) TO for(j...) for(i...) for a performance speedup

        j = jSTART;

        for(j; j < jEND + 1; j++) {

            element(i, j) = 1; // Current domain constituted by one element

        }

    }

    /* TODO - The following was to be placed in a function outside the main, but because of the complexity of the problem and debug purposes, we have not had time to create the function. It should be easy, though */

    /* COMPUTE - Fill point coordinates matrices */

    /* Fill X and Y matrices with distances point-to-axes */

    i = iSTART;

    for(i; i < iEND + 1; i++) {        

        j = jSTART;

        for(j; j < jEND + 1; j++) {

            X(i, j) =  dX*i;

            Y(i, j) = dY*j;

        }

    }

    /* Pass coordinates to the halo in order to compute geometry related variables */

    HALOUPDATE(xCollection, 1);
    HALOUPDATE(yCollection, 2);

    /* TODO - The following was to be placed in a function outside the main, but because of the complexity of the problem and debug purposes, we have not had time to create the function. It should be easy, though */

    /* COMPUTE - Fill coefficient matrices */

    /* Inner points: compute domain as though conduction was only happening. Boundary points will be replaced later */

    i = iSTART;

    for(i; i < iEND + 1; i++) {

        j = jSTART;

        for(j; j < jEND + 1; j++) {
            
            id = element(i, j);

            k = conductionHeatTransferCoef(id); // Assign according to material
            alpha = convectionHeatTransferCoef(id); // Assign according to material
            // cp = specificHeatCoef(id); // Assign according to material - Needed for a transient problem
            // dens = density(id); // Assign according to material - Needed for a transient problem

            /* Compute sides general formula */

            HORI_SIDE = (X(i + 1, j) - X(i - 1, j)) / 2;
            VERT_SIDE = (Y(i, j + 1) - Y(i, j - 1)) / 2;

            AW(i, j) = k * VERT_SIDE / (X(i, j) - X(i - 1, j));
            AE(i, j) = k * VERT_SIDE / (X(i + 1, j) - X(i, j));
            AS(i, j) = k * HORI_SIDE / (Y(i, j) - Y(i, j - 1));
            AN(i, j) = k * HORI_SIDE / (Y(i, j + 1) - Y(i, j));
            AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j);
            BP(i, j) = 0;

        }
    }
    
    /* Outer points - Left side: replace boundary values */

    if(procROW == 0) {

        i = iSTART;
        j = jSTART;

        for(j; j < jEND + 1; j++) {
            
            id = element(i, j);

            k = conductionHeatTransferCoef(id);
            alpha = convectionHeatTransferCoef(id);

            HORI_SIDE = (X(i + 1, j) - X(i, j)) / 2;
            VERT_SIDE = (Y(i, j + 1) - Y(i, j - 1)) / 2;

            AW(i, j) = 0;
            AE(i, j) = k * VERT_SIDE / (X(i + 1, j) - X(i, j));
            AS(i, j) = k * HORI_SIDE / (Y(i, j) - Y(i, j - 1));
            AN(i, j) = k * HORI_SIDE / (Y(i, j + 1) - Y(i, j));
            AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) + alpha * VERT_SIDE;
            BP(i, j) = alpha * VERT_SIDE * TLEFT;

        }
    }

    /* Outer points - Bottom side: replace boundary values */

    if(procCOL == 0) {

        i = iSTART;
        j = jSTART;

        for(i; i < iEND + 1; i++) {
            
            id = element(i, j);

            k = conductionHeatTransferCoef(id);
            alpha = convectionHeatTransferCoef(id);

            HORI_SIDE = (X(i + 1, j) - X(i - 1, j)) / 2;
            VERT_SIDE = (Y(i, j + 1) - Y(i, j)) / 2;

            AW(i, j) = k * VERT_SIDE / (X(i, j) - X(i - 1, j));
            AE(i, j) = k * VERT_SIDE / (X(i + 1, j) - X(i, j));
            AS(i, j) = 0;
            AN(i, j) = k * HORI_SIDE / (Y(i, j + 1) - Y(i, j));
            AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) + alpha * HORI_SIDE;
            BP(i, j) = alpha * HORI_SIDE * TBOTTOM;

        }

    }

    /* Outer points - Right side: replace boundary values */

    if(procROW == NProcROW - 1) {

        i = iEND;
        j = jSTART;

        for(j; j < jEND + 1; j++) {
            
            id = element(i, j);

            k = conductionHeatTransferCoef(id);
            alpha = convectionHeatTransferCoef(id);

            HORI_SIDE = (X(i, j) - X(i - 1, j)) / 2;
            VERT_SIDE = (Y(i, j + 1) - Y(i, j - 1)) / 2;

            AW(i, j) = k * VERT_SIDE / (X(i, j) - X(i - 1, j));
            AE(i, j) = 0;
            AS(i, j) = k * HORI_SIDE / (Y(i, j) - Y(i, j - 1));
            AN(i, j) = k * HORI_SIDE / (Y(i, j + 1) - Y(i, j));
            AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) - alpha * VERT_SIDE;
            BP(i, j) = - alpha * VERT_SIDE * TRIGHT;

        }
    }

    /* Outer points - Top side: replace boundary values */

    if(procCOL == NProcCOL - 1) {

        i = iSTART;
        j = jEND;

        for(i; i < iEND + 1; i++) {
            
            id = element(i, j);

            k = conductionHeatTransferCoef(id);
            alpha = convectionHeatTransferCoef(id);

            HORI_SIDE = (X(i + 1, j) - X(i - 1, j)) / 2;
            VERT_SIDE = (Y(i, j) - Y(i, j - 1)) / 2;

            AW(i, j) = k * VERT_SIDE / (X(i, j) - X(i - 1, j));
            AE(i, j) = k * VERT_SIDE / (X(i + 1, j) - X(i, j));
            AS(i, j) = k * HORI_SIDE / (Y(i, j) - Y(i, j - 1));
            AN(i, j) = 0;
            AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) - alpha * HORI_SIDE;
            BP(i, j) = - alpha * HORI_SIDE * TTOP;

        }

    }

    /* Corner points - Bottom left: replace boundary values */

    if((procROW == 0) && (procCOL == 0)) {

        i = iSTART; // 0
        j = jSTART; // 0
            
        id = element(i, j);

        k = conductionHeatTransferCoef(id);
        alpha = convectionHeatTransferCoef(id);

        HORI_SIDE = (X(i + 1, j) - X(i, j)) / 2;
        VERT_SIDE = (Y(i, j + 1) - Y(i, j)) / 2;

        AW(i, j) = 0;
        AE(i, j) = k * VERT_SIDE / (X(i + 1, j) - X(i, j));
        AS(i, j) = 0;
        AN(i, j) = k * HORI_SIDE / (Y(i, j + 1) - Y(i, j));
        AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) + alpha * HORI_SIDE + alpha * VERT_SIDE;
        BP(i, j) = alpha * HORI_SIDE * TBOTTOM + alpha * VERT_SIDE * TLEFT;

    }

    /* Corner points - Bottom right: replace boundary values */

    if((procROW == NProcROW - 1) && (procCOL == 0)) {

        i = iEND; // M - 1
        j = jSTART; // 0
            
        id = element(i, j);

        k = conductionHeatTransferCoef(id);
        alpha = convectionHeatTransferCoef(id);

        HORI_SIDE = (X(i, j) - X(i - 1, j)) / 2;
        VERT_SIDE = (Y(i, j + 1) - Y(i, j)) / 2;

        AW(i, j) = k * VERT_SIDE / (X(i, j) - X(i - 1, j));
        AE(i, j) = 0;
        AS(i, j) = 0;
        AN(i, j) = k * HORI_SIDE / (Y(i, j + 1) - Y(i, j));
        AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) + alpha * HORI_SIDE - alpha * VERT_SIDE;
        BP(i, j) = alpha * HORI_SIDE * TBOTTOM - alpha * VERT_SIDE * TRIGHT;

    }

    /* Corner points - Top right: replace boundary values */

    if((procROW == NProcROW - 1) && (procCOL == NProcCOL - 1)) {

        i = iEND; // M - 1
        j = jEND; // N - 1
            
        id = element(i, j);

        k = conductionHeatTransferCoef(id);
        alpha = convectionHeatTransferCoef(id);

        HORI_SIDE = (X(i, j) - X(i - 1, j)) / 2;
        VERT_SIDE = (Y(i, j) - Y(i, j - 1)) / 2;

        AW(i, j) = k * VERT_SIDE / (X(i, j) - X(i - 1, j));
        AE(i, j) = 0;
        AS(i, j) = k * HORI_SIDE / (Y(i, j) - Y(i, j - 1));
        AN(i, j) = 0;
        AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) - alpha * HORI_SIDE - alpha * VERT_SIDE;
        BP(i, j) = - alpha * HORI_SIDE * TTOP - alpha * VERT_SIDE * TRIGHT;

    }

    /* Corner points - Top left: replace boundary values */

    if((procROW == 0) && (procCOL == NProcCOL - 1)) {

        i = iSTART; // 0
        j = jEND; // N - 1
            
        id = element(i, j);

        k = conductionHeatTransferCoef(id);
        alpha = convectionHeatTransferCoef(id);

        HORI_SIDE = (X(i + 1, j) - X(i, j)) / 2;
        VERT_SIDE = (Y(i, j) - Y(i, j - 1)) / 2;

        AW(i, j) = 0;
        AE(i, j) = k * VERT_SIDE / (X(i + 1, j) - X(i, j));
        AS(i, j) = k * HORI_SIDE / (Y(i, j) - Y(i, j - 1));
        AN(i, j) = 0;
        AP(i, j) = AW(i, j) + AE(i, j) + AS(i, j) + AN(i, j) - alpha * HORI_SIDE + alpha * VERT_SIDE;
        BP(i, j) = - alpha * HORI_SIDE * TTOP + alpha * VERT_SIDE * TLEFT;

    }

    /* COMPUTE - Fill tempMap matrix */

    i = iSTART;

    for(i; i < iEND + 1; i++) { // TODO - Reverse for(i..) for(j...) TO for(j...) for(i...) for a performance speedup

        j = jSTART;

        for(j; j < jEND + 1; j++) {

            T(i, j) = 300; // Random init temperature

        }

    }
    /* COMPUTE - Copy temperature map so it can be updated */

    copyArr(oldTempMap, tempMap, Nmem);

    /* TODO - Program works fine without the following. The program works fine without the following, it is only needed so the user can see the original temperature map */

    /* PRINT - Initial temperature map */

    SPEAK(printf("ORIGINAL"));

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 0); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 1); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 2); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 3); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    /* COMPUTE - Solve the problem */

    for(ite; ite < iteMax + 1; ite++) { // Maximum number of iterations

        HALOUPDATE(tempMap, 0); // Update halo with most recent values

        diff = 0; // Set diff

        SOLVERJACOBI(tempMap, oldTempMap, ae, aw, an, as, ap, bp); // Apply equation

        /* Is converging? */

        compareIdx = 0;

        for(compareIdx; compareIdx < Nmem*Nmem + 1; compareIdx++) {

            diffMax = ABSVALOF(tempMap[compareIdx] - oldTempMap[compareIdx]);

            if(diffMax > diff) diff = diffMax;

        }
        
        // SPEAK_PROC(printf("diff = %f", diff), proc);

        if(diff <= delta) converged = 1; // Raise converging flag
        else copyArr(oldTempMap, tempMap, Nmem); // Update temperature matrices to compute again with more precision

       /* We could have also forced the proc to stop just after computing for an error less than delta, but in this case the procs don't stop until the proc with the largest error has not converged */

        if(MPI_Reduce(&converged, &convergedSum, 1, MPI_INT, MPI_SUM, mainProc, MPI_COMM_WORLD) == 4) break; // Stop after all processors have computed a value with an error less than delta

    }

    // SPEAK(printf("Ended computation after %i iterations. convergedSum: %i (if 4 -> all procs have converged)", ite - 1, convergedSum)) // Computation info;

    /* TODO - Save temperature map in a file. The program works fine without the following, it is only needed so the user can see the final temperature map */

    /* PRINT - Initial temperature map */

    SPEAK(printf("UPDATED"));

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 0); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 1); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 2); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    SPEAK_PROC(printer(tempMap, M, N, proc, procROW, procCOL, NProcROW, NProcCOL, iSTART, iEND, jSTART, jEND, Nmem), 3); // Check results

    checkr(MPI_Barrier(MPI_COMM_WORLD),"MPI_Barrier");

    /* PROG - Free memory */

    free(numMap);
    free(tempMap);
    free(oldTempMap);
    free(as);
    free(an);
    free(aw);
    free(ae);
    free(ap);
    free(bp);
    free(xCollection);
    free(yCollection);

    /* MPI - End */

    MPI_Finalize();

    /* EXIT */

    exit(0);
}
