pipeline {
    agent any // No global agent, each stage will define its own
    environment {
        DOCKER_CONFIG = '/tmp/.docker'  // Set to a directory with write access
        repoUri = "875986301930.dkr.ecr.eu-west-1.amazonaws.com/greenai"
        repoRegistryUrl = "875986301930.dkr.ecr.eu-west-1.amazonaws.com"
        registryCreds = 'ecr:eu-west-1:awscreds'
        cluster = "GreenCluster"
        service = "green-svc"
        region = 'eu-west-2'
    }
    stages {
        stage('Docker Test') {
            agent {
                docker {
                    image 'docker:latest'
                    args '-v /var/run/docker.sock:/var/run/docker.sock'  // Mount Docker socket
                }
            }
            steps {
                script {
                    sh 'docker ps'
                }
            }
            post {
                success {
                    echo 'Docker Test stage completed successfully.'
                }
                failure {
                    echo 'Docker Test stage failed.'
                }
                always {
                    echo 'Docker Test stage has finished.'
                }
            }
        }

        /*
        stage('Build Docker Image') {
            agent {
                docker {
                    image 'docker:latest'
                    args '-v /var/run/docker.sock:/var/run/docker.sock'  // Mount Docker socket
                }
            }
            steps {
                script {
                    echo 'Building Docker Image from Dockerfile...'
                    sh 'mkdir -p /tmp/.docker'  // Ensure the directory exists
                    dockerImage = docker.build(repoUri + ":$BUILD_NUMBER")
                }
            }
            post {
                success {
                    echo 'Docker image built successfully.'
                }
                failure {
                    echo 'Failed to build Docker image.'
                }
                always {
                    echo 'The Build Docker Image stage has finished.'
                }
            }
        }

        stage('Push Docker Image to ECR') {
            agent {
                docker {
                    image 'docker:latest'
                    args '-v /var/run/docker.sock:/var/run/docker.sock'  // Mount Docker socket
                }
            }
            steps {
                script {
                    echo "Pushing Docker Image to ECR..."
                    docker.withRegistry(repoRegistryUrl, registryCreds) {
                        dockerImage.push("$BUILD_NUMBER")
                        dockerImage.push('latest')
                    }
                }
            }
            post {
                success {
                    echo 'Docker image pushed to ECR successfully.'
                }
                failure {
                    echo 'Failed to push Docker image to ECR.'
                }
                always {
                    echo 'Push Docker Image to ECR stage has finished.'
                }
            }
        }

        stage('Deploy to ECS') {
            agent {
                docker {
                    image 'amazon/aws-cli:latest'  // Use a pre-built AWS CLI Docker image for ECS deployment
                    args '-v /var/run/docker.sock:/var/run/docker.sock --entrypoint=""'
                }
            }
            steps {
                script {
                    echo "Deploying Image to ECS..."
                    withAWS(credentials: 'awscreds', region: "${region}") {
                        sh 'aws ecs update-service --cluster ${cluster} --service ${service} --force-new-deployment'
                    }
                }
            }
            post {
                success {
                    echo 'Deployment to ECS succeeded.'
                }
                failure {
                    echo 'Deployment to ECS failed.'
                }
                always {
                    echo 'Deploy to ECS stage has finished.'
                }
            }
        }
        */
    }
}
