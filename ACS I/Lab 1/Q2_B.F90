    
      ! ==============================================      
      ! Question 2 B
      ! ==============================================
      PROGRAM Q2_B
        INTEGER          N, NRHS, KL, KU
        PARAMETER        ( N = 100, NRHS = 1, KL = 2, KU =2)
        INTEGER          LDA, LDB
        PARAMETER        ( LDA = N, LDB = N )
        INTEGER          INFO
        INTEGER          IPIV( N )
        
        INTEGER LDAB
        PARAMETER (LDAB = 2*(KL+KU+1))
        DOUBLE PRECISION A( LDAB,N), B( LDB, NRHS),  C( LDB, NRHS)
        DOUBLE PRECISION AB(LDB,NRHS), Bk( LDB, NRHS)

        EXTERNAL DGBTRF, DGBTRS
          
        DO i=1,LDAB
            DO J=1,N
                A(i,j) = 1.0
                IF(i == 1 .or. i == 2) THEN 
                    A(i,j)=0.0
                ELSE IF (i == 3) THEN
                    A(i,j) = 2
                ELSE IF (i == 4) THEN
                    A(i,j) = -1.0
                ELSE IF (i == 5) THEN
                    A(i,j) = 8.0
                ELSE IF (i == 6) THEN
                    A(i,j) = 1.0
                ELSE IF (i == 7) THEN
                    A(i,j) = 3.0
                END IF
             END DO
        END DO
        
        A(KL+i,1) = 0
        A(KL+2,1) = 0
        A(KL+1,KL) = 0
        A(LDAB,N) = 0
        A(LDAB,N-1) = 0
        A(LDAB-1,N) = 0


        DO i = 1,N
            if(i == 1) THEN
                B(i,1) = 9
            ELSE IF (i == 2) THEN
                B(i,1) = 10
            ELSE IF ( i == N-1) THEN
                B(i,1) = 11
            ELSE IF ( i == N) THEN
                B(i,1) = 12
            ELSE
                B(i,1) = 13
            END IF
        END DO
        

        WRITE(*,*)'Value of A'

        DO i = 1, LDAB
            write(*,'(1X,*(F8.3))') (A(i,j), j=1,N)
        END DO     

        WRITE(*,*)'Value of B'

        DO i = 1, LDAB
            write(*,'(1X,*(F8.3))') (B(i,1))
        END DO     

        CALL DGBTRF(N, N,KL,KU,A,LDAB, IPIV,INFO)
        
        WRITE(*,*)'Value of X'
        DO i = 1, LDAB
            write(*,'(1X,*(F8.3))') (A(i,j), j=1,N)
        END DO
        
        Bk = B
        DO k = 1,5
            
            CALL DGBTRS( 'N', N,KL, KU, NRHS, A, LDAB, IPIV, Bk, LDB, INFO )

            WRITE(*,*)'Values of Y'
            DO i=1,N
                WRITE(*,*) Bk(i,1) 
            END DO

            Bk = B + Bk
        END DO

  
        
    
        END