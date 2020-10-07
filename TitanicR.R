library('stringr')
library('dplyr')
library('tidyr')
library('caret')
library('caretEnsemble')
install.packages("bartMachine")
install.packages('rJava')
Sys.getenv("JAVA_HOME")
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_251')
library('rJava')
library('bartMachine')
library('ada')
unlink('C:\\Users\\Declan\\Documents\\R\\win-library\\3.6//00LOCK', recursive = TRUE)

train = read.csv('file:///C:/Users/Declan/Downloads/train.csv')
test = read.csv('file:///C:/Users/Declan/Downloads/test.csv')
PassengerId = test$PassengerId 
dat = dplyr::bind_rows(train,test)
head(dat)

for (i in 1:(nrow(dat['Name']))){
  #Title
  dat$Title[i] = str_extract_all(dat['Name'][i,1], ' ([a-zA-Z]+)\\.')
  #Fam Size
  dat$FamSize[i] = dat$SibSp[i] + dat$Parch[i] + 1
  dat$Alone[i] = ifelse(dat$FamSize[i]==1,1,0)
  
}

dat$Title = replace(dat$Title, which(dat$Title==c(' Lady', ' Countess',' Capt', ' Col',
                            ' Don', ' Dr', ' Major', ' Rev', ' Sir', ' Jonkheer', ' Dona')), 'Rare')

tits = c(' Lady.', ' Countess.',' Capt.', ' Col.',
  ' Don.', ' Dr.', ' Major.', ' Rev.', ' Sir.', ' Jonkheer.', ' Dona.')
for (i in 1:length(tits)){ 
  index = which(dat$Title==tits[i])
  dat$Title[index] = 'Rare'
}
dat$Title = replace(dat$Title, which(dat$Title==' Mlle.'),' Miss.')
dat$Title = replace(dat$Title, which(dat$Title==' Mme.'), ' Mrs.')
dat$Title = replace(dat$Title, which(dat$Title==' Ms.'), ' Miss.')
dat$Title[514] = ' Mrs.'
dat$Title = cbind(dat$Title)
tit = c(' Mr.', ' Mrs.', ' Miss.', 'Rare', ' Master.')

for (i in 1:length(tit)){
  index = which(dat$Title==tit[i])
  dat$Title[index,] = i

}
dat$Title = as.numeric(dat$Title)
dat$Title = dat$Title - 1

#Changes cabin to has Cabin or not
dat$Cabin = ifelse(dat$Cabin=="",0,1)
#Fill NAs in Embarked
dat$Embarked= tidyr::replace_na(dat$Embarked,'S')
dat$Embarked[which(dat$Embarked=="")] = 'S'
dat$Embarked = as.factor(dat$Embarked)
dat$Embarked = as.numeric(dat$Embarked) -1
#Ticket levels 
dat$Fare = replace(dat$Fare,which(is.na(dat$Fare)), median(dat$Fare, na.rm=TRUE))
dat$FareLevels = ntile(dat$Fare,4)
dat$Fare = as.factor(dat$FareLevels)
dat$Fare = as.numeric(dat$Fare)
dat$Fare = dat$Fare -1 

#Age: Impute random age values within 1 SD of mean 
NAs = which(is.na(dat$Age))
avg = mean(dat$Age[-NAs])
sd = sd(dat$Age[-NAs])
num_of_ages = length(which(is.na(dat$Age)))
ages = rnorm(num_of_ages, mean=avg, sd=sd/2)
dat$Age[NAs] = ages
dat$Age = as.integer(dat$Age)
#Age Levels 
#dat$Agen[20] = ntile(dat$Age,5)
dat$Age = cut(dat$Age,5)
dat$Age = as.numeric(dat$Age)
dat$Age = dat$Age -1

dat = select(dat, select =-c('PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamSize','FareLevels'))
train = dat[1:891,]
test = dat[-(1:891),-1]




cont = trainControl(method='cv', number = 10, savePredictions = 'final',
                    summaryFunction = twoClassSummary, classProbs = T, index = createFolds(train$Survived,10))

train$Survived = as.factor(ifelse(train$Survived==0,'No','Yes') )

modelLookup('adaboost')
gbm_grid = expand.grid(n.trees = 600,
                       shrinkage = 0.04,
                       interaction.depth = 6,
                       n.minobsinnode = 10)
rf_grid = expand.grid(.mtry = 2:6 )

ada_grid = expand.grid(nIter= seq(40,250,10), method ='AdaBoost.M1') 

rpart_grid = expand.grid(cp = seq(0, 0.05, 0.005))

knn_grid = expand.grid(k = 2:15)

mlp_grid = expand.grid(size = 1:5)

ada = train(Survived~.,data=train, tuneGrid = ada_grid, method ='adaboost')
rf =  caret::train(Survived~., data=train, method='rf', tuneGrid =rf_grid)
gbm = train(Survived~., data=train, method='gbm', tuneGrid =gbm_grid)
knn = train(Survived~., data=train, method='knn', tuneGrid =knn_grid)
mlp = train(Survived~., data=train, method='mlp', tuneGrid =mlp_grid)
rpart = train(Survived~., data=train, method='rpart', tuneGrid =rpart_grid)

tune_list = list(rf = caretModelSpec(method='rf', tuneGrid=data.frame(.mtry=3)),
                 gbm = caretModelSpec(method='gbm', tuneGrid=gbm_grid),
                 ada = caretModelSpec(method='ada'),
                 rpart = caretModelSpec(method='rpart', cp=0.015),
                 knn = caretModelSpec(method='knn', tuneGrid= data.frame(k=6)),
                 mlp = caretModelSpec(method='mlp', tuneGrid=data.frame(size = 4))) 


model_list = caretList(Survived~.,data=train, trControl = cont,
                       methodList = c('rf','adaboost','gbm',
                      'rpart','knn','mlp') , metric='ROC')#,continue_on_fail=T)
model_list = caretList(Survived~.,data=train, trControl = cont,
                       tuneList = tune_list , metric='ROC')

modelCor(resamples(model_list))

ens1 =caretEnsemble::caretStack(model_list, method = 'xgbDART')
summary(ens1)
p1 = predict(ens1, newdata=test,type='prob')
p1 = ifelse(p1>=0.5,0,1)

preds = cbind(PassengerId,p1)
preds = as.data.frame(preds)
preds$Survived = preds$p1
preds = preds[,c(1,3)]
write.csv(preds,file='file:///C:/Users/Declan/TitanicR.csv', row.names =FALSE)
 
          