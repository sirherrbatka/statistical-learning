(cl:in-package #:statistical-learning.gradient-boost-tree)


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters fundamental-gradient-boost-tree-parameters)
                                            &rest initargs
                                            &key attributes train-data
                                              target-data weights response
                                              expected-value shrinkage data-points
                                            &allow-other-keys)
  (declare (ignore initargs))
  (sl.mp:make-training-state/proxy
   parameters/proxy
   (implementation parameters
                   :shrinkage shrinkage
                   :expected-value expected-value)
   :data-points data-points
   :train-data (sl.data:wrap train-data)
   :target-data (if (null response)
                    (target parameters
                            (sl.data:wrap target-data)
                            (sl.data:wrap expected-value))
                    response)
   :attributes attributes
   :weights weights))


(defmethod sl.mp:make-model*/proxy
    (parameters-proxy
     (parameters gradient-boosting-implementation)
     state)
  (let ((parameters (sl.mp:training-parameters state)))
    (make 'gradient-boost-model
          :parameters (gradient-parameters parameters)
          :shrinkage (shrinkage parameters)
          :expected-value (expected-value parameters)
          :root (sl.tp:root (call-next-method)))))


(defmethod sl.mp:make-model*/proxy
    (parameters/proxy
     (parameters fundamental-gradient-boost-tree-parameters)
     state)
  (sl.mp:make-model* (sl.mp:training-parameters state) state))


(defmethod sl.tp:contribute-predictions*/proxy
    (parameters/proxy
     (parameters regression)
     model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
  (ensure leaf-key #'identity)
  (sl.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (contributed-predictions parameters model data-points-count)))
    (let* ((sums (sl.tp:sums state))
           (weight (sl.tp:weight model))
           (splitter (sl.tp:splitter parameters))
           (shrinkage (shrinkage model))
           (root (sl.tp:root model)))
      (declare (type sl.data:double-float-data-matrix sums))
      (sl.data:data-matrix-map-data-points
       (lambda (data-point d)
         (declare (ignore d))
         (let* ((leaf (~>> (sl.tp:leaf-for splitter root
                                           data data-point
                                           model)
                           (funcall leaf-key)))
                (predictions (sl.tp:predictions leaf)))
           (iterate
             (for i from 0 below (array-dimension predictions 0))
             (incf (sl.data:mref sums data-point i)
                   (* shrinkage
                      (aref predictions 0 i)
                      weight)))))
       sums
       parallel)
      (incf (sl.tp:contributions-count state) weight))
    state))


(defmethod sl.tp:contribute-predictions*/proxy
    (parameters/proxy
     (parameters classification)
     model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
  (ensure leaf-key #'identity)
  (sl.data:bind-data-matrix-dimensions ((data-points-count attributes-count data))
    (when (null state)
      (setf state (contributed-predictions parameters model data-points-count)))
    (let* ((sums (sl.tp:sums state))
           (weight (sl.tp:weight model))
           (splitter (sl.tp:splitter parameters))
           (number-of-classes (sl.opt:number-of-classes parameters))
           (shrinkage (shrinkage model))
           (root (sl.tp:root model)))
      (sl.data:data-matrix-map-data-points
       (lambda (data-point d)
         (declare (ignore d))
         (iterate
           (declare (type fixnum j))
           (with leaf = (~>> (sl.tp:leaf-for splitter root
                                             data data-point
                                             model)
                             (funcall leaf-key)))
           (with predictions = (sl.tp:predictions leaf))
           (for j from 0 below number-of-classes)
           (for gradient = (aref predictions 0 j))
           (incf (sl.data:mref sums data-point j)
                 (* weight shrinkage gradient))))
       sums
       parallel)
      (incf (sl.tp:contributions-count state) weight))
    state))


(defmethod sl.tp:extract-predictions*/proxy
    (parameters/proxy
     (parameters regression)
     (state sl.tp:contributed-predictions))
   (~> state sl.tp:sums))


(defmethod sl.tp:extract-predictions*/proxy
    (parameters/proxy
     (parameters classification)
     (state sl.tp:contributed-predictions))
  (iterate
    (declare (type fixnum i number-of-classes)
             (optimize (speed 3) (safety 0))
             (type double-float maximum sum)
             (type sl.data:double-float-data-matrix sums result))
    (with optimized-function = (optimized-function parameters))
    (with number-of-classes = (sl.opt:number-of-classes optimized-function))
    (with sums = (sl.tp:sums state))
    (with result = (sl.data:make-data-matrix-like sums))
    (for i from 0 below (sl.data:data-points-count sums))
    (for maximum = most-negative-double-float)
    (for sum = 0.0d0)
    (iterate
      (declare (type fixnum j))
      (for j from 0 below number-of-classes)
      (maxf maximum (sl.data:mref sums i j)))
    (iterate
      (declare (type fixnum j)
               (type double-float out))
      (for j from 0 below number-of-classes)
      (for out = (exp (- (sl.data:mref sums i j) maximum)))
      (setf (sl.data:mref result i j) out)
      (incf sum out))
    (iterate
      (declare (type fixnum j))
      (for j from 0 below number-of-classes)
      (setf #1=(sl.data:mref result i j) (/ #1# sum)))
    (finally (return result))))


(defmethod implementation
    ((parameters classification) &rest initargs)
  (apply #'make 'classification-implementation
         :proxy (sl.common:proxy parameters)
         :gradient-parameters parameters
         :splitter (sl.tp:splitter parameters)
         :maximal-depth (sl.tp:maximal-depth parameters)
         :minimal-difference (sl.tp:minimal-difference parameters)
         :minimal-size (sl.tp:minimal-size parameters)
         :parallel (sl.tp:parallel parameters)
         initargs))


(defmethod implementation ((parameters regression) &rest initargs)
  (apply #'make 'regression-implementation
         :proxy (sl.common:proxy parameters)
         :gradient-parameters parameters
         :splitter (sl.tp:splitter parameters)
         :maximal-depth (sl.tp:maximal-depth parameters)
         :minimal-difference (sl.tp:minimal-difference parameters)
         :minimal-size (sl.tp:minimal-size parameters)
         :parallel (sl.tp:parallel parameters)
         initargs))


(defmethod target ((parameters classification)
                   target-data expected-value)
  (declare (type sl.data:double-float-data-matrix target-data expected-value)
           (optimize (speed 3) (safety 0)
                     (space 0) (debug 0)
                     (compilation-speed 0)))
  (iterate
    (declare (type sl.data:double-float-data-matrix result)
             (type fixnum i number-of-classes
                   data-points-count target))
    (with optimized-function = (optimized-function parameters))
    (with number-of-classes = (sl.opt:number-of-classes optimized-function))
    (with data-points-count = (sl.data:data-points-count target-data))
    (with result = (sl.data:make-data-matrix data-points-count
                                             number-of-classes))
    (for i from 0 below data-points-count)
    (for target = (truncate (sl.data:mref target-data i 0)))
    (iterate
      (declare (type fixnum j))
      (for j from 0 below number-of-classes)
      (setf (sl.data:mref result i j)
            (- (if (= target j) 1 0)
               (sl.data:mref expected-value 0 j))))
    (finally (return result))))


(defmethod target ((parameters regression)
                   target-data expected-value)
  (declare (optimize (speed 3) (safety 0)
                     (space 0) (debug 0)
                     (compilation-speed 0))
           (type sl.data:double-float-data-matrix target-data expected-value))
  (iterate
    (declare (type sl.data:double-float-data-matrix result)
             (type fixnum i))
    (with result = (sl.data:make-data-matrix-like target-data))
    (for i from 0 below (sl.data:data-points-count result))
    (iterate
      (declare (type fixnum ii))
      (for ii from 0 below (sl.data:attributes-count result))
      (setf (sl.data:mref result i ii)
            (- (the double-float (sl.data:mref target-data i ii))
               (the double-float (sl.data:mref expected-value 0 ii)))))
    (finally (return result))))


(defmethod sl.tp:initialize-leaf/proxy
    (parameters/proxy
     (training-parameters classification-implementation)
     training-state
     leaf)
  (iterate
    (declare (type fixnum length number-of-classes)
             (type sl.data:double-float-data-matrix target-data)
             (type (simple-array double-float (1 *)) result))
    (with target-data = (sl.mp:target-data training-state))
    (with length = (sl.data:data-points-count target-data))
    (with number-of-classes = (sl.data:attributes-count target-data))
    (with result = (make-array `(1 ,number-of-classes) :element-type 'double-float))
    (for j from 0 below length)
    (iterate
      (declare (type fixnum j))
      (for k from 0 below number-of-classes)
      (incf (aref result 0 k)
            (sl.data:mref target-data j k)))
    (finally (setf (sl.tp:predictions leaf) (sl.data:map-array
                                             (lambda (x) (/ x length))
                                             result)))))


(defmethod calculate-expected-value ((parameters classification)
                                     data)
  (declare (type sl.data:double-float-data-matrix data)
           (optimize (speed 3) (safety 0)
                     (space 0) (debug 0)
                     (compilation-speed 0)))
  (iterate
    (declare (type fixnum i)
             (type (simple-array double-float (* *)) result))
    (with number-of-classes = (~>> parameters
                                   optimized-function
                                   sl.opt:number-of-classes))
    (with result = (make-array `(1 ,number-of-classes) :element-type 'double-float))
    (for i from 0 below (sl.data:data-points-count data))
    (incf (aref result 0
                (truncate (sl.data:mref data i 0))))
    (finally
     (iterate
       (declare (type fixnum j))
       (for j from 0 below number-of-classes)
       (for avg = (/ #1=(aref result 0 j)
                     (sl.data:data-points-count data)))
       (setf #1# avg))
     (return (sl.data:wrap result)))))


(defmethod calculate-expected-value ((parameters regression) data)
  (iterate
    (declare (type fixnum i)
             (optimize (speed 3) (safety 0)
                       (space 0) (debug 0)
                       (compilation-speed 0))
             (type sl.data:double-float-data-matrix result))
    (with number-of-attributes = (sl.data:attributes-count data))
    (with result = (sl.data:make-data-matrix 1 number-of-attributes))
    (for i from 0 below (sl.data:data-points-count data))
    (iterate
      (for ii from 0 below number-of-attributes)
      (incf (sl.data:mref result 0 ii)
            (sl.data:mref data i ii)))
    (finally
     (iterate
       (declare (type fixnum j))
       (for j from 0 below number-of-attributes)
       (for avg = (/ #1=(sl.data:mref result 0 j)
                     (sl.data:data-points-count data)))
       (setf #1# avg))
     (return result))))


(defmethod calculate-response ((parameters regression)
                               gathered-predictions
                               expected)
  (~>> (sl.tp:extract-predictions gathered-predictions)
       (sl.opt:response (optimized-function parameters)
                        expected)
       sl.data:wrap))


(defmethod calculate-response ((parameters classification)
                               gathered-predictions
                               expected)
  (~>> (sl.tp:sums gathered-predictions)
       (sl.opt:response (optimized-function parameters)
                        expected)
       sl.data:wrap))


(defmethod sl.opt:number-of-classes ((object classification))
  (~> object optimized-function sl.opt:number-of-classes))


(defmethod contributed-predictions ((parameters regression)
                                    model
                                    data-points-count)
  (make 'sl.tp:contributed-predictions
        :training-parameters parameters
        :sums (iterate
                (with expected-value = (expected-value model))
                (with attributes-count =
                      (sl.data:attributes-count expected-value))
                (with result = (sl.data:make-data-matrix data-points-count
                                                         attributes-count))
                (for i from 0 below data-points-count)
                (iterate
                  (for ii from 0 below attributes-count)
                  (setf (sl.data:mref result i ii)
                        (sl.data:mref expected-value 0 ii)))
                (finally (return result)))))


(defmethod contributed-predictions ((parameters classification)
                                    model
                                    data-points-count)
  (make 'sl.tp:contributed-predictions
        :training-parameters parameters
        :sums (iterate
                (declare (type sl.data:double-float-data-matrix result))
                (with number-of-classes = (sl.opt:number-of-classes parameters))
                (with result = (sl.data:make-data-matrix data-points-count
                                                         number-of-classes))
                (with expected-value = (expected-value model))
                (for i from 0 below data-points-count)
                (iterate
                  (for j from 0 below number-of-classes)
                  (setf (sl.data:mref result i j) (sl.data:mref expected-value 0 j)))
                (finally (return result)))))


(defmethod cl-ds.utils:cloning-information
    append ((object fundamental-gradient-boost-tree-parameters))
  '((:optimized-function optimized-function)))
