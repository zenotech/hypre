/*
 * File:          Hypre_StructGrid_IOR.h
 * Symbol:        Hypre.StructGrid-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:22 PDT
 * Generated:     20020904 10:05:24 PDT
 * Description:   Intermediate Object Representation for Hypre.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_StructGrid_IOR_h
#define included_Hypre_StructGrid_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.StructGrid" (version 0.1.5)
 * 
 * Define a structured grid class.
 */

struct Hypre_StructGrid__array;
struct Hypre_StructGrid__object;

extern struct Hypre_StructGrid__object*
Hypre_StructGrid__new(void);

extern struct Hypre_StructGrid__object*
Hypre_StructGrid__remote(const char *url);

extern void Hypre_StructGrid__init(
  struct Hypre_StructGrid__object* self);
extern void Hypre_StructGrid__fini(
  struct Hypre_StructGrid__object* self);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_StructGrid__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_StructGrid__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_StructGrid__object* self);
  void (*f__ctor)(
    struct Hypre_StructGrid__object* self);
  void (*f__dtor)(
    struct Hypre_StructGrid__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.5.1 */
  void (*f_addReference)(
    struct Hypre_StructGrid__object* self);
  void (*f_deleteReference)(
    struct Hypre_StructGrid__object* self);
  SIDL_bool (*f_isInstanceOf)(
    struct Hypre_StructGrid__object* self,
    const char* name);
  SIDL_bool (*f_isSame)(
    struct Hypre_StructGrid__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    struct Hypre_StructGrid__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.5.1 */
  /* Methods introduced in Hypre.StructGrid-v0.1.5 */
  int32_t (*f_Assemble)(
    struct Hypre_StructGrid__object* self);
  int32_t (*f_SetCommunicator)(
    struct Hypre_StructGrid__object* self,
    void* MPI_comm);
  int32_t (*f_SetDimension)(
    struct Hypre_StructGrid__object* self,
    int32_t dim);
  int32_t (*f_SetExtents)(
    struct Hypre_StructGrid__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper);
  int32_t (*f_SetPeriodic)(
    struct Hypre_StructGrid__object* self,
    struct SIDL_int__array* periodic);
};

/*
 * Define the class object structure.
 */

struct Hypre_StructGrid__object {
  struct SIDL_BaseClass__object d_sidl_baseclass;
  struct Hypre_StructGrid__epv* d_epv;
  void*                         d_data;
};

/*
 * Create a dense array of the given dimension with specified
 * index bounds.  This array owns and manages its data.
 * All object pointers are initialized to NULL.
 */

struct Hypre_StructGrid__array*
Hypre_StructGrid__iorarray_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/*
 * Create an array that uses data memory from another source.
 * This initial contents are determined by the data being
 * borrowed.
 */

struct Hypre_StructGrid__array*
Hypre_StructGrid__iorarray_borrow(
  struct Hypre_StructGrid__object** firstElement,
  int32_t                           dimen,
  const int32_t                     lower[],
  const int32_t                     upper[],
  const int32_t                     stride[]);

/*
 * Destroy the given array. Trying to destroy a NULL array is a
 * noop.
 */

void
Hypre_StructGrid__iorarray_destroy(
  struct Hypre_StructGrid__array* array);

/*
 * Return the number of dimensions in the array. If the
 * array pointer is NULL, zero is returned.
 */

int32_t
Hypre_StructGrid__iorarray_dimen(const struct Hypre_StructGrid__array *array);

/*
 * Return the lower bound on dimension ind. If ind is not
 * a valid dimension, zero is returned.
 */

int32_t
Hypre_StructGrid__iorarray_lower(const struct Hypre_StructGrid__array *array,
  int32_t ind);

/*
 * Return the upper bound on dimension ind. If ind is not
 * a valid dimension, negative one is returned.
 */

int32_t
Hypre_StructGrid__iorarray_upper(const struct Hypre_StructGrid__array *array,
  int32_t ind);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_StructGrid__object*
Hypre_StructGrid__iorarray_get4(
  const struct Hypre_StructGrid__array* array,
  int32_t                               i1,
  int32_t                               i2,
  int32_t                               i3,
  int32_t                               i4);

/*
 * Get an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the return value is non-NULL, the client owns one
 * reference to the object/interface. The client must
 * decrement the reference count when done with the reference.
 */

struct Hypre_StructGrid__object*
Hypre_StructGrid__iorarray_get(
  const struct Hypre_StructGrid__array* array,
  const int32_t                         indices[]);

/*
 * Set an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the incoming value is non-NULL, this function will increment
 * the reference code of the object/interface. If it is
 * overwriting a non-NULL pointer, the reference count of the
 * object/interface being overwritten will be decremented.
 */

void
Hypre_StructGrid__iorarray_set4(
  struct Hypre_StructGrid__array*  array,
  int32_t                          i1,
  int32_t                          i2,
  int32_t                          i3,
  int32_t                          i4,
  struct Hypre_StructGrid__object* value);

/*
 * Set an element of a multi-dimensional array. This will use
 * the indices provided up to the actual dimension of the array.
 * The values of excess indices are ignored.
 * 
 * If the incoming value is non-NULL, this function will increment
 * the reference code of the object/interface. If it is
 * overwriting a non-NULL pointer, the reference count of the
 * object/interface being overwritten will be decremented.
 */

void
Hypre_StructGrid__iorarray_set(
  struct Hypre_StructGrid__array*  array,
  const int32_t                    indices[],
  struct Hypre_StructGrid__object* value);

struct Hypre_StructGrid__external {
  struct Hypre_StructGrid__object*
  (*createObject)(void);

  struct Hypre_StructGrid__object*
  (*createRemote)(const char *url);

  struct Hypre_StructGrid__array*
  (*createArray)(
    int32_t       dimen,
    const int32_t lower[],
    const int32_t upper[]);

  struct Hypre_StructGrid__array*
  (*borrowArray)(
    struct Hypre_StructGrid__object** firstElement,
    int32_t                           dimen,
    const int32_t                     lower[],
    const int32_t                     upper[],
    const int32_t                     stride[]);

  void
  (*destroyArray)(
    struct Hypre_StructGrid__array* array);

  int32_t
  (*getDimen)(const struct Hypre_StructGrid__array *array);

  int32_t
  (*getLower)(const struct Hypre_StructGrid__array *array, int32_t ind);

  int32_t
  (*getUpper)(const struct Hypre_StructGrid__array *array, int32_t ind);

  struct Hypre_StructGrid__object*
  (*getElement)(
    const struct Hypre_StructGrid__array* array,
    const int32_t                         indices[]);

  struct Hypre_StructGrid__object*
  (*getElement4)(
    const struct Hypre_StructGrid__array* array,
    int32_t                               i1,
    int32_t                               i2,
    int32_t                               i3,
    int32_t                               i4);

  void
  (*setElement)(
    struct Hypre_StructGrid__array*  array,
    const int32_t                    indices[],
    struct Hypre_StructGrid__object* value);
void
(*setElement4)(
  struct Hypre_StructGrid__array*  array,
  int32_t                          i1,
  int32_t                          i2,
  int32_t                          i3,
  int32_t                          i4,
  struct Hypre_StructGrid__object* value);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructGrid__external*
Hypre_StructGrid__externals(void);

#ifdef __cplusplus
}
#endif
#endif
