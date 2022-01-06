use sales;

-- drop table new_train;
-- create table new_train
-- select * from train where year(date) between 2015 and 2017;

-- create table sales_by_store_MY
-- select store_nbr, DATE_FORMAT(date,'%Y-%m') as month_year, round(sum(sales),2) as total_sales 
-- from new_train 
-- group by DATE_FORMAT(date,'%Y-%m'), store_nbr
-- order by store_nbr, DATE_FORMAT(date,'%Y-%m');


-- since sales is measured in different units, it's only significant if we consider 
--   
-- create table product
-- select DATE_FORMAT(t1.date,'%Y-%m') as month_year, family, round(sum(sales),2) as sales, 
-- 	sum(transactions) as transactions
-- from train t1 left join transactions t2 on t1.date = t2.date and t1.store_nbr = t2.store_nbr
-- group by month_year, family;

-- select dayofmonth(date), avg(transactions), '2013' as year from transactions 
-- where year(date) = 2013
-- group by dayofmonth(date)
-- union
-- select dayofmonth(date), avg(transactions), '2014' as year from transactions 
-- where year(date) = 2014
-- group by dayofmonth(date)
-- union
-- select dayofmonth(date), avg(transactions), '2015' as year from transactions 
-- where year(date) = 2015
-- group by dayofmonth(date)
-- union
-- select dayofmonth(date), avg(transactions), '2016' as year from transactions 
-- where year(date) = 2016
-- group by dayofmonth(date)
-- union
-- select dayofmonth(date), avg(transactions), '2017' as year  from transactions 
-- where year(date) = 2017
-- group by dayofmonth(date);

select avg(transactions) as transactions, avg(sales) as sales
from train t1 left join transactions t2 using(date)
group by DATE_FORMAT(t1.date,'%Y-%m');



