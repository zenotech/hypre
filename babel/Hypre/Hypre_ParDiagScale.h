/*
 * File:          Hypre_ParDiagScale.h
 * Symbol:        Hypre.ParDiagScale-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:22 PDT
 * Generated:     20020904 10:05:27 PDT
 * Description:   Client-side glue code for Hypre.ParDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_ParDiagScale_h
#define included_Hypre_ParDiagScale_h

/**
 * Symbol "Hypre.ParDiagScale" (version 0.1.5)
 */
struct Hypre_ParDiagScale__object;
struct Hypre_ParDiagScale__array;
typedef struct Hypre_ParDiagScale__object* Hypre_ParDiagScale;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Operator_h
#include "Hypre_Operator.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
Hypre_ParDiagScale
Hypre_ParDiagScale__create(void);

/**
 * Method:  SetLogging
 */
int32_t
Hypre_ParDiagScale_SetLogging(
  Hypre_ParDiagScale self,
  int32_t level);

/**
 * Method:  Setup
 */
int32_t
Hypre_ParDiagScale_Setup(
  Hypre_ParDiagScale self,
  Hypre_Vector x,
  Hypre_Vector y);

/**
 * Method:  SetIntArrayParameter
 */
int32_t
Hypre_ParDiagScale_SetIntArrayParameter(
  Hypre_ParDiagScale self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Method:  SetIntParameter
 */
int32_t
Hypre_ParDiagScale_SetIntParameter(
  Hypre_ParDiagScale self,
  const char* name,
  int32_t value);

/**
 * Method:  GetResidual
 */
int32_t
Hypre_ParDiagScale_GetResidual(
  Hypre_ParDiagScale self,
  Hypre_Vector* r);

/**
 * Method:  SetPrintLevel
 */
int32_t
Hypre_ParDiagScale_SetPrintLevel(
  Hypre_ParDiagScale self,
  int32_t level);

/**
 * Method:  GetIntValue
 */
int32_t
Hypre_ParDiagScale_GetIntValue(
  Hypre_ParDiagScale self,
  const char* name,
  int32_t* value);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_ParDiagScale_SetCommunicator(
  Hypre_ParDiagScale self,
  void* comm);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_ParDiagScale_isInstanceOf(
  Hypre_ParDiagScale self,
  const char* name);

/**
 * Method:  SetStringParameter
 */
int32_t
Hypre_ParDiagScale_SetStringParameter(
  Hypre_ParDiagScale self,
  const char* name,
  const char* value);

/**
 * Method:  SetDoubleParameter
 */
int32_t
Hypre_ParDiagScale_SetDoubleParameter(
  Hypre_ParDiagScale self,
  const char* name,
  double value);

/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteReference</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
SIDL_BaseInterface
Hypre_ParDiagScale_queryInterface(
  Hypre_ParDiagScale self,
  const char* name);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_ParDiagScale_isSame(
  Hypre_ParDiagScale self,
  SIDL_BaseInterface iobj);

/**
 * Method:  Apply
 */
int32_t
Hypre_ParDiagScale_Apply(
  Hypre_ParDiagScale self,
  Hypre_Vector x,
  Hypre_Vector* y);

/**
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */
void
Hypre_ParDiagScale_addReference(
  Hypre_ParDiagScale self);

/**
 * Method:  GetDoubleValue
 */
int32_t
Hypre_ParDiagScale_GetDoubleValue(
  Hypre_ParDiagScale self,
  const char* name,
  double* value);

/**
 * Method:  SetOperator
 */
int32_t
Hypre_ParDiagScale_SetOperator(
  Hypre_ParDiagScale self,
  Hypre_Operator A);

/**
 * Method:  SetDoubleArrayParameter
 */
int32_t
Hypre_ParDiagScale_SetDoubleArrayParameter(
  Hypre_ParDiagScale self,
  const char* name,
  struct SIDL_double__array* value);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_ParDiagScale_deleteReference(
  Hypre_ParDiagScale self);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_ParDiagScale
Hypre_ParDiagScale__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_ParDiagScale__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_ParDiagScale__array*
Hypre_ParDiagScale__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_ParDiagScale__array*
Hypre_ParDiagScale__array_borrow(
  struct Hypre_ParDiagScale__object** firstElement,
  int32_t                             dimen,
  const int32_t                       lower[],
  const int32_t                       upper[],
  const int32_t                       stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_ParDiagScale__array_destroy(
  struct Hypre_ParDiagScale__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_ParDiagScale__array_dimen(const struct Hypre_ParDiagScale__array *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_ParDiagScale__array_lower(const struct Hypre_ParDiagScale__array *array,
  int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_ParDiagScale__array_upper(const struct Hypre_ParDiagScale__array *array,
  int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_ParDiagScale__object*
Hypre_ParDiagScale__array_get(
  const struct Hypre_ParDiagScale__array* array,
  const int32_t                           indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_ParDiagScale__object*
Hypre_ParDiagScale__array_get4(
  const struct Hypre_ParDiagScale__array* array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_ParDiagScale__array_set(
  struct Hypre_ParDiagScale__array*  array,
  const int32_t                      indices[],
  struct Hypre_ParDiagScale__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_ParDiagScale__array_set4(
  struct Hypre_ParDiagScale__array*  array,
  int32_t                            i1,
  int32_t                            i2,
  int32_t                            i3,
  int32_t                            i4,
  struct Hypre_ParDiagScale__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_ParDiagScale__array_get1(a,i1) \
  Hypre_ParDiagScale__array_get4(a,i1,0,0,0)

#define Hypre_ParDiagScale__array_get2(a,i1,i2) \
  Hypre_ParDiagScale__array_get4(a,i1,i2,0,0)

#define Hypre_ParDiagScale__array_get3(a,i1,i2,i3) \
  Hypre_ParDiagScale__array_get4(a,i1,i2,i3,0)

#define Hypre_ParDiagScale__array_set1(a,i1,v) \
  Hypre_ParDiagScale__array_set4(a,i1,0,0,0,v)

#define Hypre_ParDiagScale__array_set2(a,i1,i2,v) \
  Hypre_ParDiagScale__array_set4(a,i1,i2,0,0,v)

#define Hypre_ParDiagScale__array_set3(a,i1,i2,i3,v) \
  Hypre_ParDiagScale__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
