(cl:in-package #:cl-user)


(defpackage #:covtype-example
  (:use #:cl #:cl-grf.aux-package))

(in-package #:covtype-example)

(ql:quickload :vellum)

(defparameter *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :cl-grf)
                                  (merge-pathnames "examples/covtype.data"))
                        :header nil)
      (vellum:to-table :columns '((:alias elevation :type integer)
                                  (:alias aspect :type integer)
                                  (:alias slope :type integer)
                                  (:alias Horizontal_Distance_To_Hydrology :type integer)
                                  (:alias Vertical_Distance_To_Hydrology :type integer)
                                  (:alias Horizontal_Distance_To_Roadways :type integer)
                                  (:alias Hillshade_9am :type integer)
                                  (:alias Hillshade_Noon :type integer)
                                  (:alias Hillshade_3pm :type integer)
                                  (:alias Horizontal_Distance_To_Fire_Points :type integer)
                                  (:alias Wilderness_Area_1 :type integer)
                                  (:alias Wilderness_Area_2 :type integer)
                                  (:alias Wilderness_Area_3 :type integer)
                                  (:alias Wilderness_Area_4 :type integer)
                                  (:alias Soil_Type_1 :type integer)
                                  (:alias Soil_Type_2 :type integer)
                                  (:alias Soil_Type_3 :type integer)
                                  (:alias Soil_Type_4 :type integer)
                                  (:alias Soil_Type_5 :type integer)
                                  (:alias Soil_Type_6 :type integer)
                                  (:alias Soil_Type_7 :type integer)
                                  (:alias Soil_Type_8 :type integer)
                                  (:alias Soil_Type_9 :type integer)
                                  (:alias Soil_Type_10 :type integer)
                                  (:alias Soil_Type_11 :type integer)
                                  (:alias Soil_Type_12 :type integer)
                                  (:alias Soil_Type_13 :type integer)
                                  (:alias Soil_Type_14 :type integer)
                                  (:alias Soil_Type_15 :type integer)
                                  (:alias Soil_Type_16 :type integer)
                                  (:alias Soil_Type_17 :type integer)
                                  (:alias Soil_Type_18 :type integer)
                                  (:alias Soil_Type_19 :type integer)
                                  (:alias Soil_Type_20 :type integer)
                                  (:alias Soil_Type_21 :type integer)
                                  (:alias Soil_Type_22 :type integer)
                                  (:alias Soil_Type_23 :type integer)
                                  (:alias Soil_Type_24 :type integer)
                                  (:alias Soil_Type_25 :type integer)
                                  (:alias Soil_Type_26 :type integer)
                                  (:alias Soil_Type_27 :type integer)
                                  (:alias Soil_Type_28 :type integer)
                                  (:alias Soil_Type_29 :type integer)
                                  (:alias Soil_Type_30 :type integer)
                                  (:alias Soil_Type_31 :type integer)
                                  (:alias Soil_Type_32 :type integer)
                                  (:alias Soil_Type_33 :type integer)
                                  (:alias Soil_Type_34 :type integer)
                                  (:alias Soil_Type_35 :type integer)
                                  (:alias Soil_Type_36 :type integer)
                                  (:alias Soil_Type_37 :type integer)
                                  (:alias Soil_Type_38 :type integer)
                                  (:alias Soil_Type_39 :type integer)
                                  (:alias Soil_Type_40 :type integer)
                                  (:alias Cover_Type :type integer)))))


(defparameter *train-data*
  (vellum:as-matrix (vellum:select *data*
                      ;; :rows '(:take-to 500)
                      :columns '(:take-to soil_type_40))
                    'double-float))


(defparameter *cover-types*
  (vellum:with-table (*data*)
    (bind (((min . max) (cl-ds.alg:extrema *data* #'< :key
                                           (vellum:brr cover_type))))
      (assert (= min 1))
      (assert (= max 7))
      (1+ (- max min)))))


(defparameter *target-data*
  (vellum:as-matrix (vellum:select *data*
                      ;; :rows '(:take-to 500)
                      :columns '(:v cover_type)
                      )
                    'double-float))

(iterate
  (for i from 0 below (cl-grf.data:data-points-count *target-data*))
  (decf (cl-grf.data:mref *target-data* i 0)))


(defparameter *training-parameters*
  (make 'cl-grf.algorithms:single-information-gain-classification
        :maximal-depth 16
        :minimal-difference 0.0000000000000001d0
        :number-of-classes *cover-types*
        :minimal-size 5
        :trials-count 5000
        :parallel nil))


(defparameter *forest-parameters*
  (make 'cl-grf.forest:random-forest-parameters
        :trees-count 500
        :forest-class 'cl-grf.forest:classification-random-forest
        :parallel t
        :tree-attributes-count 30
        :tree-sample-size 200
        :tree-parameters *training-parameters*))

(defparameter *model*
  (cl-grf.mp:make-model *forest-parameters*
                        *train-data*
                        *target-data*))

(defparameter *predictions*
  (cl-grf.mp:predict *model*
                     (~> (vellum:select *data*
                           :columns '(:take-to soil_type_40)
                           :rows '(:take-to 5))
                         (vellum:as-matrix 'double-float))))
