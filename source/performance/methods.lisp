(cl:in-package #:cl-grf.performance)


(defmethod print-object ((object confusion-matrix) stream)
  (print-unreadable-object (object stream :type t)
    (format stream
            "Precision: ~$% Recall: ~$% Accuracy: ~$% Specificity ~$%"
            (* 100 (precision object))
            (* 100 (recall object))
            (* 100 (accuracy object))
            (* 100 (specificity object))))
  object)


(defmethod cl-ds.utils:cloning-information append ((object confusion-matrix))
  '((:positive positive)
    (:negative negative)
    (:true-positive true-positive)
    (:true-negative true-negative)))


(defmethod shared-initialize :after ((object confusion-matrix)
                                     slots
                                     &rest initargs)
  (declare (ignore slots initargs))
  (let ((true-positive (true-positive object))
        (positive (positive object))
        (negative (negative object))
        (true-negative (true-negative object)))
    (unless (integerp true-positive)
      (error 'type-error
             :expect-type 'integer
             :datum true-positive))
    (unless (integerp true-negative)
      (error 'type-error
             :expect-type 'integer
             :datum true-negative))
    (unless (integerp negative)
      (error 'type-error
             :expect-type 'integer
             :datum negative))
    (unless (integerp positive)
      (error 'type-error
             :expect-type 'integer
             :datum positive))
    (unless (>= positive true-positive)
      (error 'cl-ds:incompatible-arguments
             :parameters '(:positive :true-positive)
             :values (list positive true-positive)
             :format-control "TRUE-POSITIVE cannot be larger then POSITIVE."))
    (unless (>= negative true-negative)
      (error 'cl-ds:incompatible-arguments
             :parameters '(:negative :true-negative)
             :values (list negative true-negative)
             :format-control "TRUE-NEGATIVE cannot be larger then NEGATIVE."))))


(defmethod performance-metric :around ((parameters fundamental-model-parameters)
                                       target predictions)
  (cl-grf.data:check-data-points target)
  (check-type predictions vector)
  (unless (= (cl-grf.data:data-points-count target)
             (length predictions))
    (error 'cl-ds:incompatible-arguments
           :parameters '(target predictions)
           :values (list target predictions)
           :format-control "TARGET and PREDICITIONS are supposed to contain equal number of data-points"))
  (call-next-method parameters target predictions))
