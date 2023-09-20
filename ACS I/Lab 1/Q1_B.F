    
      ! ==============================================      
      ! Question 1 B
      ! ==============================================
      PROGRAM Q1_B
      INTEGER          N, NRHS
      PARAMETER        ( N = 2, NRHS = 1 )
      INTEGER          LDA, LDB
      PARAMETER        ( LDA = N, LDB = N )
      INTEGER          INFO
      INTEGER          IPIV( N )
      DOUBLE PRECISION A( LDA, N), B( LDB, NRHS),  C( LDB, NRHS), X(2,1)
      EXTERNAL DGESV
        
      DATA  A/
     $ 4.5, 3.1,
     $ 1.6, 1.1
     $  /

      DATA  B/
     $ 19.249,
     $ 6.843
     $  /

      DATA  C/
     $ 19.25,
     $ 6.84
     $  /

    !   DO i=1,N 
    !      X(i,1) = (dble(i) / (N))
    !   END DO

    !   B = MATMUL(A,X)

      CALL DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      WRITE(*,*)'Value of X'
      DO i=1,2
            WRITE(*,*) B(i,1) 
      END DO


      DATA  A/
     $ 4.5, 3.1,
     $ 1.6, 1.1
     $  /

    !   DO i=1,N 
    !     X(i,1) = (dble(i) / (N))
    !   END DO

    !   C = MATMUL(A,X)

      CALL DGESV( N, NRHS, A, LDA, IPIV, C, LDB, INFO )
      WRITE(*,*)'Values of Y'
      DO i=1,2
            WRITE(*,*) C(i,1) 
      END DO
  
      END