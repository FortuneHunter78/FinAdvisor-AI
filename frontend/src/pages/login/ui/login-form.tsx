import type { FC } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/shared/ui/form';
import { Input } from '@/shared/ui/input';
import {
  loginFormFields,
  LoginFormSchema,
  loginFormSchemaType,
} from '../lib/form-schema';
import LoadingButton from '@/shared/ui/loading-button';
import PasswordInput from '@/shared/ui/password-input';

const LoginForm: FC = () => {
  const form = useForm<loginFormSchemaType>({
    resolver: zodResolver(LoginFormSchema),
  });

  function onSubmit(values: loginFormSchemaType) {
    console.log(values);
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)}>
        {loginFormFields.map(({ name, label, type }) => (
          <FormField
            key={name}
            control={form.control}
            name={name}
            render={({ field }) => (
              <FormItem className='mt-5'>
                <FormLabel>{label}</FormLabel>
                <FormControl>
                  {type === 'password' ? (
                    <PasswordInput placeholder={label} {...field} type={type} />
                  ) : (
                    <Input placeholder={label} {...field} type={type} />
                  )}
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        ))}

        <LoadingButton type='submit' size='lg' className='w-full mt-8'>
          Войти
        </LoadingButton>
      </form>
    </Form>
  );
};

export default LoginForm;
