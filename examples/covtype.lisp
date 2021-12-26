(ql:quickload '(:statistical-learning :vellum :vellum-csv))

(cl:in-package #:cl-user)

(defpackage #:covtype-example
  (:use #:cl #:statistical-learning.aux-package))

(in-package #:covtype-example)

(defvar *data*
  (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                              (merge-pathnames "examples/covtype.data"))
                    :includes-header-p nil
                    :columns '((:name elevation :type integer)
                               (:name aspect :type integer)
                               (:name slope :type integer)
                               (:name Horizontal_Distance_To_Hydrology :type number)
                               (:name Vertical_Distance_To_Hydrology :type number)
                               (:name Horizontal_Distance_To_Roadways :type number)
                               (:name Hillshade_9am :type integer)
                               (:name Hillshade_Noon :type integer)
                               (:name Hillshade_3pm :type integer)
                               (:name Horizontal_Distance_To_Fire_Points :type integer)
                               (:name Wilderness_Area_1 :type integer)
                               (:name Wilderness_Area_2 :type integer)
                               (:name Wilderness_Area_3 :type integer)
                               (:name Wilderness_Area_4 :type integer)
                               (:name Soil_Type_1 :type integer)
                               (:name Soil_Type_2 :type integer)
                               (:name Soil_Type_3 :type integer)
                               (:name Soil_Type_4 :type integer)
                               (:name Soil_Type_5 :type integer)
                               (:name Soil_Type_6 :type integer)
                               (:name Soil_Type_7 :type integer)
                               (:name Soil_Type_8 :type integer)
                               (:name Soil_Type_9 :type integer)
                               (:name Soil_Type_10 :type integer)
                               (:name Soil_Type_11 :type integer)
                               (:name Soil_Type_12 :type integer)
                               (:name Soil_Type_13 :type integer)
                               (:name Soil_Type_14 :type integer)
                               (:name Soil_Type_15 :type integer)
                               (:name Soil_Type_16 :type integer)
                               (:name Soil_Type_17 :type integer)
                               (:name Soil_Type_18 :type integer)
                               (:name Soil_Type_19 :type integer)
                               (:name Soil_Type_20 :type integer)
                               (:name Soil_Type_21 :type integer)
                               (:name Soil_Type_22 :type integer)
                               (:name Soil_Type_23 :type integer)
                               (:name Soil_Type_24 :type integer)
                               (:name Soil_Type_25 :type integer)
                               (:name Soil_Type_26 :type integer)
                               (:name Soil_Type_27 :type integer)
                               (:name Soil_Type_28 :type integer)
                               (:name Soil_Type_29 :type integer)
                               (:name Soil_Type_30 :type integer)
                               (:name Soil_Type_31 :type integer)
                               (:name Soil_Type_32 :type integer)
                               (:name Soil_Type_33 :type integer)
                               (:name Soil_Type_34 :type integer)
                               (:name Soil_Type_35 :type integer)
                               (:name Soil_Type_36 :type integer)
                               (:name Soil_Type_37 :type integer)
                               (:name Soil_Type_38 :type integer)
                               (:name Soil_Type_39 :type integer)
                               (:name Soil_Type_40 :type integer)
                               (:name Cover_Type :type integer))
                    :body (vellum:bind-row (cover_type horizontal_distance_to_roadways)
                            (decf cover_type))))

(defvar *cover-types* ; 7
  (vellum:with-table (*data*)
    (bind (((min . max) (cl-ds.alg:extrema *data* #'< :key (vellum:brr cover_type))))
      (assert (zerop min))
      (1+ (- max min)))))

(defvar *train-data*
  (vellum:to-matrix (vellum:select *data*
                      :columns (vellum:s (vellum:between :to 'cover_type)))
                    :element-type 'double-float))

(defvar *target-data*
  (vellum:to-matrix (vellum:select *data*
                      :columns '(cover_type))
                    :element-type 'double-float))

(defparameter *training-parameters*
  (make 'statistical-learning.dt:classification
        :optimized-function (sl.opt:gini-impurity *cover-types*)
        :maximal-depth 30
        :minimal-difference 0.0001d0
        :minimal-size 10
        :splitter (sl.common:lift (make-instance 'sl.tp:random-attribute-splitter)
                                  'sl.tp:random-splitter
                                  :trials-count 50)
        :parallel t))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 1000
        :parallel t
        :pruning (make-instance 'sl.club-drf:club-drf
                                :number-of-trees-selected 200
                                :sample-size 200
                                :parallel t
                                :max-neighbor 20)
        :weights-calculator (make-instance 'sl.ensemble:dynamic-weights-calculator)
        :tree-batch-size 10
        :tree-attributes-count 45
        :data-points-sampler (make-instance 'sl.ensemble:weights-based-data-points-sampler
                                            :sampling-rate 0.15)
        :tree-parameters *training-parameters*))

(lparallel:check-kernel)
(defparameter *confusion-matrix*
  (statistical-learning.performance:cross-validation *forest-parameters*
                                                     2
                                                     *train-data*
                                                     *target-data*
                                                     :parallel t))

(format t "~3$~%" (statistical-learning.performance:accuracy *confusion-matrix*)) ; ~0.83

(~> (make 'statistical-learning.ensemble:gradient-boost-ensemble
          :trees-count 250
          :parallel t
          :tree-batch-size 5
          :tree-attributes-count 50
          :shrinkage 0.35d0
          :data-points-sampler (make 'sl.ensemble:gradient-based-one-side-sampler
                                     :small-gradient-sampling-rate 0.1
                                     :large-gradient-sampling-rate 0.1)
          :tree-parameters (make 'sl.gbt:classification
                                 :optimized-function (sl.opt:k-logistic *cover-types*)
                                 :maximal-depth 25
                                 :minimal-size 20
                                 :minimal-difference 0.00001d0
                                 :parallel t
                                 :splitter (sl.common:lift (make-instance 'sl.tp:random-attribute-splitter)
                                                           'sl.tp:random-splitter
                                                           :trials-count 50)))
    (statistical-learning.performance:cross-validation 2
                                                       *train-data*
                                                       *target-data*
                                                       :parallel t)
    statistical-learning.performance:accuracy
    (format t "~3$~%" _)) ; ~0.90
